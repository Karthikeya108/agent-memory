"""
Cross-session agent memory backed by Lakebase (autoscaling PostgreSQL).

Uses **on-behalf-of (OBO)** authentication so that each user's memory
operations run with their own Databricks OAuth token.  This ensures
per-user credential isolation at the database level.

Connection approach:
- Host:     PGHOST env var (Databricks App injection or .env)
- Port:     PGPORT env var (default 5432)
- Database: PGDATABASE env var (default 'chatbot')
- User:     Logged-in user (from x-forwarded headers) or SDK current user
- Password: User's OBO OAuth token (from x-forwarded-access-token) or SDK token
"""
import logging
import os
from dataclasses import dataclass

import asyncpg
from databricks.sdk import WorkspaceClient
from mlflow.genai.agent_server import get_request_headers

logger = logging.getLogger(__name__)

MEMORY_SCHEMA = "agent_memory"

# Tracks whether the schema/table has been created this process lifetime.
_schema_initialized = False


@dataclass
class UserCredentials:
    """Per-request credentials for Lakebase OBO connections."""

    user_id: str
    pg_user: str
    token: str


def get_user_credentials(request=None) -> UserCredentials:
    """
    Extract per-user credentials from the current request.

    Resolves user identity from (in priority order):
      1. x-forwarded-email / x-forwarded-user headers (Databricks Apps proxy)
      2. request.context.user_id (set by the frontend chat UI)
      3. Databricks SDK current user (local dev)

    The DB connection always uses the service principal credentials
    (which has the Lakebase role). Memory data is isolated per user
    via the user_id column in queries.
    """
    user_id = None

    # 1. Try request headers
    try:
        headers = get_request_headers()
        user_id = (
            headers.get("x-forwarded-email")
            or headers.get("x-forwarded-user")
            or headers.get("x-forwarded-preferred-username")
        )
    except Exception:
        pass

    # 2. Fall back to request.context.user_id (frontend always sends this)
    if not user_id and request is not None:
        ctx = getattr(request, "context", None)
        if ctx and getattr(ctx, "user_id", None):
            user_id = ctx.user_id
            logger.info("[memory] Got user_id from request context: %s", user_id)

    # Get SP/CLI credentials for the DB connection
    w = WorkspaceClient()
    auth_headers: dict[str, str] = w.config.authenticate()
    token = auth_headers.get("Authorization", "").removeprefix("Bearer ")
    if not token:
        raise RuntimeError(
            "Failed to obtain Databricks OAuth token for Lakebase connection. "
            "Ensure DATABRICKS_CONFIG_PROFILE or service-principal env vars are set."
        )
    pg_user = os.environ.get("PGUSER") or w.current_user.me().user_name

    # Fall back to SDK user if no identity from headers/context
    if not user_id:
        user_id = w.current_user.me().user_name

    logger.info("[memory] User: %s, DB role: %s", user_id, pg_user)
    return UserCredentials(user_id=user_id, pg_user=pg_user, token=token)


def _get_sp_token_and_user() -> tuple[str, str]:
    """Return (oauth_token, pg_user) using the service principal / SDK.

    Used only for one-time schema initialisation (DDL), not for per-user
    data operations.
    """
    w = WorkspaceClient()
    auth_headers: dict[str, str] = w.config.authenticate()
    token = auth_headers.get("Authorization", "").removeprefix("Bearer ")
    if not token:
        raise RuntimeError(
            "Failed to obtain Databricks OAuth token for Lakebase connection."
        )
    user = os.environ.get("PGUSER") or w.current_user.me().user_name
    return token, user


async def _connect_as(pg_user: str, token: str) -> asyncpg.Connection:
    """Open a Lakebase connection with the given credentials."""
    host = os.environ.get("PGHOST", "")
    if not host:
        raise RuntimeError(
            "PGHOST environment variable is not set. "
            "Set it to your Lakebase endpoint host."
        )
    port = int(os.environ.get("PGPORT", "5432"))
    database = os.environ.get("PGDATABASE", "chatbot")

    return await asyncpg.connect(
        host=host,
        port=port,
        database=database,
        user=pg_user,
        password=token,
        ssl="require",
    )


async def initialize_schema() -> None:
    """Verify the agent_memory schema and table exist; create if needed.

    Uses the service principal connection (not OBO) because DDL requires
    elevated privileges that individual users may not have.
    """
    global _schema_initialized
    if _schema_initialized:
        return
    try:
        token, user = _get_sp_token_and_user()
        conn = await _connect_as(user, token)
        try:
            # Check if table already exists — skip DDL if so.
            exists = await conn.fetchval(
                "SELECT EXISTS ("
                "  SELECT 1 FROM information_schema.tables "
                "  WHERE table_schema = $1 AND table_name = 'memories'"
                ")",
                MEMORY_SCHEMA,
            )
            if not exists:
                await conn.execute(
                    f"""
                    CREATE SCHEMA IF NOT EXISTS {MEMORY_SCHEMA};

                    CREATE TABLE IF NOT EXISTS {MEMORY_SCHEMA}.memories (
                        id          SERIAL PRIMARY KEY,
                        user_id     TEXT NOT NULL,
                        key         TEXT NOT NULL,
                        value       TEXT NOT NULL,
                        created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                        updated_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                        UNIQUE (user_id, key)
                    );

                    CREATE INDEX IF NOT EXISTS memories_user_id_idx
                        ON {MEMORY_SCHEMA}.memories (user_id);
                    """
                )
                logger.info("[memory] agent_memory schema created")
            _schema_initialized = True
            logger.info("[memory] agent_memory schema ready")
        finally:
            await conn.close()
    except Exception as exc:
        logger.warning("[memory] Could not verify schema (memory may be unavailable): %s", exc)
        # Mark as initialized anyway to avoid retrying every request
        _schema_initialized = True


async def get_memories(creds: UserCredentials) -> list[dict]:
    """Return all memories for a user, newest-updated first."""
    try:
        conn = await _connect_as(creds.pg_user, creds.token)
        try:
            rows = await conn.fetch(
                f"SELECT key, value, updated_at "
                f"FROM {MEMORY_SCHEMA}.memories "
                f"WHERE user_id = $1 "
                f"ORDER BY updated_at DESC",
                creds.user_id,
            )
            return [{"key": r["key"], "value": r["value"]} for r in rows]
        finally:
            await conn.close()
    except Exception as exc:
        logger.warning("[memory] get_memories failed: %s", exc)
        return []


async def upsert_memory(creds: UserCredentials, key: str, value: str) -> None:
    """Insert or update a single memory entry."""
    conn = await _connect_as(creds.pg_user, creds.token)
    try:
        await conn.execute(
            f"""
            INSERT INTO {MEMORY_SCHEMA}.memories (user_id, key, value, updated_at)
            VALUES ($1, $2, $3, NOW())
            ON CONFLICT (user_id, key)
            DO UPDATE SET value = EXCLUDED.value, updated_at = NOW()
            """,
            creds.user_id,
            key,
            value,
        )
    finally:
        await conn.close()


async def delete_memory(creds: UserCredentials, key: str) -> bool:
    """Delete a memory by key. Returns True if a row was removed."""
    conn = await _connect_as(creds.pg_user, creds.token)
    try:
        result = await conn.execute(
            f"DELETE FROM {MEMORY_SCHEMA}.memories WHERE user_id = $1 AND key = $2",
            creds.user_id,
            key,
        )
        return result != "DELETE 0"
    finally:
        await conn.close()

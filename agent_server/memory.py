"""
Cross-session agent memory backed by Lakebase (autoscaling PostgreSQL).

Provides three layers of memory:
  1. Key-value facts (memories table) — explicit user preferences
  2. Conversation turn embeddings (conversation_embeddings) — per-turn vector storage
  3. Conversation summaries (conversation_summaries) — per-chat summary vectors

Layers 2 & 3 use pgvector for semantic similarity search, allowing the agent
to recall relevant context from past conversations when the user explicitly asks.

Connection approach:
- Host:     PGHOST env var (Databricks App injection or .env)
- Port:     PGPORT env var (default 5432)
- Database: PGDATABASE env var (default 'chatbot')
- User:     SP credentials for DB connection; user isolation via user_id column
"""
import json
import logging
import os
from dataclasses import dataclass

import asyncpg
from databricks.sdk import WorkspaceClient
from databricks_openai import AsyncDatabricksOpenAI
from mlflow.genai.agent_server import get_request_headers

logger = logging.getLogger(__name__)

MEMORY_SCHEMA = "agent_memory"
CHAT_SCHEMA = "ai_chatbot"

# Semantic search configuration
EMBEDDING_MODEL = "databricks-gte-large-en"
EMBEDDING_DIMENSION = 1024
SIMILARITY_THRESHOLD = 0.6
TOP_K_TURNS = 5
TOP_K_SUMMARIES = 3

# Tracks whether schemas have been created this process lifetime.
_schema_initialized = False
_semantic_schema_initialized = False


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


async def get_chat_history(conversation_id: str, creds: UserCredentials) -> list[dict]:
    """Load previous messages for a conversation from the ai_chatbot schema.

    Returns user/assistant text messages in chronological order, suitable
    for prepending to the current request's input.
    """
    try:
        conn = await _connect_as(creds.pg_user, creds.token)
        try:
            rows = await conn.fetch(
                f'SELECT role, parts FROM {CHAT_SCHEMA}."Message" '
                f'WHERE "chatId" = $1 '
                f'ORDER BY "createdAt" ASC',
                conversation_id,
            )
            messages = []
            for r in rows:
                role = r["role"]
                if role not in ("user", "assistant"):
                    continue
                parts = r["parts"]  # JSON column
                # Extract text content from parts
                if isinstance(parts, list):
                    if isinstance(parts[0], str):
                        parts = [json.loads(p) if isinstance(p, str) else p for p in parts]
                    for part in parts:
                        if isinstance(part, dict) and part.get("type") == "text" and part.get("text"):
                            messages.append({
                                "role": role,
                                "content": part["text"],
                            })
                            break  # one text block per message
            return messages
        finally:
            await conn.close()
    except Exception as exc:
        logger.warning("[memory] get_chat_history failed: %s", exc)
        return []


# ---------------------------------------------------------------------------
# Semantic conversation memory (pgvector)
# ---------------------------------------------------------------------------


async def initialize_semantic_schema() -> None:
    """Create pgvector extension and semantic memory tables if needed."""
    global _semantic_schema_initialized
    if _semantic_schema_initialized:
        return
    try:
        token, user = _get_sp_token_and_user()
        conn = await _connect_as(user, token)
        try:
            exists = await conn.fetchval(
                "SELECT EXISTS ("
                "  SELECT 1 FROM information_schema.tables "
                "  WHERE table_schema = $1 AND table_name = 'conversation_embeddings'"
                ")",
                MEMORY_SCHEMA,
            )
            if not exists:
                await conn.execute(
                    f"""
                    CREATE EXTENSION IF NOT EXISTS vector;

                    CREATE TABLE IF NOT EXISTS {MEMORY_SCHEMA}.conversation_embeddings (
                        id              SERIAL PRIMARY KEY,
                        user_id         TEXT NOT NULL,
                        chat_id         TEXT NOT NULL,
                        turn_index      INTEGER NOT NULL,
                        content_text    TEXT NOT NULL,
                        embedding       vector({EMBEDDING_DIMENSION}) NOT NULL,
                        created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
                    );

                    CREATE INDEX IF NOT EXISTS conv_emb_user_idx
                        ON {MEMORY_SCHEMA}.conversation_embeddings (user_id);

                    CREATE INDEX IF NOT EXISTS conv_emb_vector_idx
                        ON {MEMORY_SCHEMA}.conversation_embeddings
                        USING hnsw (embedding vector_cosine_ops);

                    CREATE TABLE IF NOT EXISTS {MEMORY_SCHEMA}.conversation_summaries (
                        id              SERIAL PRIMARY KEY,
                        user_id         TEXT NOT NULL,
                        chat_id         TEXT NOT NULL UNIQUE,
                        title           TEXT,
                        summary_text    TEXT NOT NULL,
                        key_topics      TEXT[],
                        embedding       vector({EMBEDDING_DIMENSION}) NOT NULL,
                        message_count   INTEGER NOT NULL DEFAULT 0,
                        created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                        updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
                    );

                    CREATE INDEX IF NOT EXISTS conv_sum_user_idx
                        ON {MEMORY_SCHEMA}.conversation_summaries (user_id);

                    CREATE INDEX IF NOT EXISTS conv_sum_vector_idx
                        ON {MEMORY_SCHEMA}.conversation_summaries
                        USING hnsw (embedding vector_cosine_ops);
                    """
                )
                logger.info("[semantic] Semantic memory tables created")
            _semantic_schema_initialized = True
            logger.info("[semantic] Semantic memory schema ready")
        finally:
            await conn.close()
    except Exception as exc:
        logger.warning("[semantic] Could not initialize semantic schema: %s", exc)
        _semantic_schema_initialized = True


async def generate_embedding(text: str) -> list[float]:
    """Generate a vector embedding via the Databricks Foundation Model API."""
    client = AsyncDatabricksOpenAI()
    response = await client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=text[:8000],  # truncate to stay within model limits
    )
    return response.data[0].embedding


def _embedding_to_pg(embedding: list[float]) -> str:
    """Format an embedding list as a pgvector literal string."""
    return "[" + ",".join(str(v) for v in embedding) + "]"


async def store_turn_embedding(
    creds: UserCredentials,
    chat_id: str,
    turn_index: int,
    content_text: str,
    embedding: list[float],
) -> None:
    """Store a turn-pair embedding for later similarity search."""
    conn = await _connect_as(creds.pg_user, creds.token)
    try:
        await conn.execute(
            f"""
            INSERT INTO {MEMORY_SCHEMA}.conversation_embeddings
                (user_id, chat_id, turn_index, content_text, embedding)
            VALUES ($1, $2, $3, $4, $5::vector)
            """,
            creds.user_id,
            chat_id,
            turn_index,
            content_text,
            _embedding_to_pg(embedding),
        )
    finally:
        await conn.close()


async def store_conversation_summary(
    creds: UserCredentials,
    chat_id: str,
    title: str | None,
    summary_text: str,
    key_topics: list[str],
    embedding: list[float],
    message_count: int,
) -> None:
    """Insert or update a conversation summary with its embedding."""
    conn = await _connect_as(creds.pg_user, creds.token)
    try:
        await conn.execute(
            f"""
            INSERT INTO {MEMORY_SCHEMA}.conversation_summaries
                (user_id, chat_id, title, summary_text, key_topics, embedding, message_count, updated_at)
            VALUES ($1, $2, $3, $4, $5, $6::vector, $7, NOW())
            ON CONFLICT (chat_id)
            DO UPDATE SET
                summary_text = EXCLUDED.summary_text,
                key_topics = EXCLUDED.key_topics,
                embedding = EXCLUDED.embedding,
                message_count = EXCLUDED.message_count,
                updated_at = NOW()
            """,
            creds.user_id,
            chat_id,
            title,
            summary_text,
            key_topics,
            _embedding_to_pg(embedding),
            message_count,
        )
    finally:
        await conn.close()


async def find_similar_conversations(
    creds: UserCredentials,
    query_embedding: list[float],
    current_chat_id: str | None = None,
) -> dict:
    """Find semantically similar past conversation context for this user.

    Searches both fine-grained turn embeddings and coarse conversation
    summaries.  Excludes the current chat to avoid self-matching.
    Returns {"turns": [...], "summaries": [...]}.
    """
    emb_str = _embedding_to_pg(query_embedding)
    exclude_chat = current_chat_id or ""

    conn = await _connect_as(creds.pg_user, creds.token)
    try:
        # Fine-grained: individual turn pairs
        turn_rows = await conn.fetch(
            f"""
            SELECT chat_id, content_text,
                   1 - (embedding <=> $1::vector) AS similarity
            FROM {MEMORY_SCHEMA}.conversation_embeddings
            WHERE user_id = $2 AND chat_id != $3
                  AND 1 - (embedding <=> $1::vector) > $4
            ORDER BY embedding <=> $1::vector
            LIMIT $5
            """,
            emb_str,
            creds.user_id,
            exclude_chat,
            SIMILARITY_THRESHOLD,
            TOP_K_TURNS,
        )

        # Coarse-grained: conversation summaries
        summary_rows = await conn.fetch(
            f"""
            SELECT chat_id, title, summary_text, key_topics,
                   1 - (embedding <=> $1::vector) AS similarity
            FROM {MEMORY_SCHEMA}.conversation_summaries
            WHERE user_id = $2 AND chat_id != $3
                  AND 1 - (embedding <=> $1::vector) > $4
            ORDER BY embedding <=> $1::vector
            LIMIT $5
            """,
            emb_str,
            creds.user_id,
            exclude_chat,
            SIMILARITY_THRESHOLD,
            TOP_K_SUMMARIES,
        )

        turns = [
            {
                "chat_id": r["chat_id"],
                "content_text": r["content_text"],
                "similarity": float(r["similarity"]),
            }
            for r in turn_rows
        ]
        summaries = [
            {
                "chat_id": r["chat_id"],
                "title": r["title"],
                "summary_text": r["summary_text"],
                "key_topics": r["key_topics"],
                "similarity": float(r["similarity"]),
            }
            for r in summary_rows
        ]
        return {"turns": turns, "summaries": summaries}
    finally:
        await conn.close()


async def generate_conversation_summary(
    messages: list[dict],
) -> tuple[str, list[str]]:
    """Use the LLM to produce a summary and topic tags for a conversation.

    Takes the last 20 messages to stay within context limits.
    Returns (summary_text, key_topics).
    """
    recent = messages[-20:]
    conversation_text = "\n".join(
        f"{m.get('role', 'unknown').title()}: {m.get('content', '')}" for m in recent
    )
    prompt = (
        "Summarize the following conversation in 2-3 sentences, "
        "capturing the key topics discussed and any conclusions reached. "
        "Also extract 3-5 key topic tags.\n\n"
        "Format your response exactly as:\n"
        "Summary: <summary>\n"
        "Topics: <comma-separated tags>\n\n"
        f"Conversation:\n{conversation_text}"
    )

    client = AsyncDatabricksOpenAI()
    response = await client.chat.completions.create(
        model="databricks-gpt-5-2",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=300,
    )
    text = response.choices[0].message.content or ""

    # Parse response
    summary = text
    topics: list[str] = []
    for line in text.splitlines():
        if line.lower().startswith("summary:"):
            summary = line.split(":", 1)[1].strip()
        elif line.lower().startswith("topics:"):
            topics = [t.strip() for t in line.split(":", 1)[1].split(",") if t.strip()]
    return summary, topics

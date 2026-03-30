# Agent Memory — Conversational Agent with Persistent Memory on Lakebase

A conversational AI agent built with the **OpenAI Agents SDK** and deployed as a **Databricks App**. The agent uses **Lakebase** (Databricks-managed PostgreSQL) to persist user-specific memories across sessions, enabling personalized, context-aware conversations.

## Architecture

```
┌─────────────────────────────────────────────────────────────-┐
│                     Databricks App                           │
│                                                              │
│  ┌──────────────────┐       ┌────────────────────────────-┐  │
│  │   Next.js Chat   │       │    FastAPI / MLflow         │  │
│  │   Frontend (UI)  │──────▶│    AgentServer (Backend)    │  │
│  │   Port 3000      │ REST  │    Port 8000                │  │
│  └──────────────────┘       │                             │  │
│                             │  ┌────────────────────────┐ │  │
│                             │  │  OpenAI Agents SDK     │ │  │
│                             │  │  (Agent, Runner,       │ │  │
│                             │  │   function_tool)       │ │  │
│                             │  └──────┬────────┬────────┘ │  │
│                             └─────────┼────────┼──────────┘  │
│                                       │        │             │
└───────────────────────────────────────┼────────┼─────────────┘
                                        │        │
                  ┌─────────────────────┘        └──────────────────┐
                  ▼                                                  ▼
    ┌──────────────────────-┐                       ┌──────────────────────────┐
    │  Databricks LLM       │                       │  Lakebase (PostgreSQL)   │
    │  (GPT-5-2 via         │                       │                          │
    │   Foundation Model    │                       │  Schema: agent_memory    │
    │   API)                │                       │  - memories (key-value)  │
    └──────────────────────-┘                       │  - conversation_         │
                                                    │    embeddings (pgvector) │
    ┌──────────────────────-┐                       │  - conversation_         │
    │  Embedding Model      │                       │    summaries (pgvector)  │
    │  (GTE-Large-EN via    │                       │                          │
    │   Foundation Model    │                       │  Schema: ai_chatbot      │
    │   API)                │                       │  - Chat, Message,        │
    └──────────────────────-┘                       │    User, Vote            │
                                                    └──────────────────────────┘
    ┌──────────────────────-┐
    │  MLflow Tracking      │
    │  (Traces, Experiments,│
    │   Evaluation)         │
    └──────────────────────-┘
```

## Per-User Data Isolation

All data is isolated per user using the **email address** as the canonical user identity across both frontend and backend:

```
Databricks Apps Reverse Proxy
  │
  ├─ X-Forwarded-Email ──► Frontend: Chat.userId = email
  │                         (chat history, messages scoped per user)
  │
  └─ request.context.user_id ──► Backend: memories.user_id = email
                                  (long-term memory scoped per user)
```

- The **frontend** (Next.js) uses `X-Forwarded-Email` from the Databricks Apps reverse proxy as the user ID for storing chat history in the `ai_chatbot` schema.
- The **backend** (agent server) extracts the user identity from `request.context.user_id` (set by the frontend) or `X-Forwarded-Email` headers.
- The **Lakebase connection** uses the app's service principal credentials. User isolation is enforced at the application layer via the `user_id` column in all queries.
- Each user only sees their own chats in the sidebar and their own memories in the agent's context.

## Key Components

| Component | File | Description |
|-----------|------|-------------|
| Agent logic | `agent_server/agent.py` | Defines the agent, its instructions, memory tools, and request handlers (`@invoke`, `@stream`) |
| Memory layer | `agent_server/memory.py` | Key-value memory, pgvector embeddings, similarity search, conversation summaries |
| Server | `agent_server/start_server.py` | FastAPI server powered by MLflow `AgentServer` with automatic tracing |
| Utilities | `agent_server/utils.py` | Session ID extraction, MCP URL building, stream event processing |
| Evaluation | `agent_server/evaluate_agent.py` | Multi-turn conversation simulation with MLflow scorers |
| App launcher | `scripts/start_app.py` | Starts both backend and frontend concurrently with health monitoring |
| Frontend | `e2e-chatbot-app-next/` | Next.js/Express chat UI with per-user chat history |
| Bundle config | `databricks.yml` | Databricks Asset Bundle definition with parameterised variables |

## Functionalities

### Persistent Memory (Lakebase-backed)

The agent can **remember**, **recall**, and **forget** facts about each user across sessions:

- **`remember(key, value)`** — Saves a fact (e.g., preferred language, name, role) to Lakebase. Upserts on conflict.
- **`recall()`** — Retrieves all stored memories for the current user. Automatically injected into the system prompt at conversation start.
- **`forget(key)`** — Deletes a specific memory when it's no longer accurate or the user requests it.

Memories are scoped per user via the `user_id` column (keyed by email address). The agent proactively saves personal details and greets returning users by their first name.

### Per-User Chat History

- Chat conversations are stored in the `ai_chatbot` schema on Lakebase
- Each chat is tied to the logged-in user's email address
- The sidebar shows only the current user's conversation history
- Chat titles are auto-generated from the user's first message

### Personalised Greeting

The agent extracts the user's first name from their email (e.g., `karthik.subbarao@databricks.com` -> **Karthik**) and greets them at the start of each conversation. If the identity is unknown (e.g., a service principal), the agent asks for the user's name.

### LLM-Powered Conversations

- Uses **GPT-5-2** via the Databricks Foundation Model API
- Supports both **streaming** and **non-streaming** responses
- Follows the [MLflow ResponsesAgent](https://mlflow.org/docs/latest/genai/flavors/responses-agent-intro/) interface (OpenAI Responses API compatible)

### Semantic Conversation Recall (pgvector)

The agent can search past conversations by topic when the user explicitly asks — powered by pgvector on Lakebase.

**How it works:**
- After every assistant response, the user+assistant turn pair is embedded via `databricks-gte-large-en` and stored in `conversation_embeddings` (background, zero latency impact)
- After 3+ turns, a conversation summary is generated by the LLM and stored in `conversation_summaries` with its embedding
- When the user explicitly asks (e.g., "what did we discuss about lakehouse?", "recall our earlier chat"), the agent calls `search_past_conversations` which:
  1. Embeds the query
  2. Runs pgvector cosine similarity search across both tables
  3. Returns matching excerpts and summaries scoped to that user only

**The agent does NOT search automatically** — only when the user explicitly requests it.

### Built-in Tools

| Tool | Description |
|------|-------------|
| `get_current_time()` | Returns the current date and time |
| `remember(key, value)` | Save a fact about the user |
| `recall()` | Retrieve all saved memories |
| `forget(key)` | Delete a specific memory |
| `search_past_conversations(query)` | Semantic search over past conversation history (pgvector) |

**MCP Server support** — Extensible with Unity Catalog function tools (e.g., `system.ai.python_exec` for code execution)

### Observability

- **MLflow autologging** for all LLM calls
- **Session tracking** via `conversation_id`
- **Experiment-linked tracing** with git-based version tracking

### Agent Evaluation

Multi-turn conversation simulation with MLflow scorers:
- Completeness, Fluency, Relevance, Safety
- Knowledge Retention, User Frustration detection
- Tool Call Correctness

## Prerequisites

- Python 3.11+
- [uv](https://docs.astral.sh/uv/getting-started/installation/) (Python package manager)
- [nvm](https://github.com/nvm-sh/nvm) with Node 20 LTS (`nvm use 20`)
- [Databricks CLI](https://docs.databricks.com/aws/en/dev-tools/cli/install)
- Access to a Databricks workspace with:
  - Foundation Model API enabled (GPT-5-2 for chat, GTE-Large-EN for embeddings)
  - A Lakebase instance with pgvector extension (for memory, chat persistence, and semantic search)

## Setup

### 1. Quick Start (Recommended)

```bash
uv run quickstart
```

This will verify dependencies, configure Databricks authentication, create an MLflow experiment, and start the app.

### 2. Manual Setup

**Configure authentication:**

```bash
databricks auth login
```

**Create your `.env` file:**

```bash
cp .env.example .env
```

Edit `.env` and fill in:

```env
DATABRICKS_CONFIG_PROFILE=dev

# MLflow experiment for tracing
MLFLOW_EXPERIMENT_ID=<your-experiment-id>

# Lakebase connection (for memory + chat history)
PGHOST=<your-lakebase-endpoint-host>
PGDATABASE=chatbot
PGPORT=5432
```

**Create an MLflow experiment:**

```bash
DATABRICKS_USERNAME=$(databricks current-user me --profile dev | jq -r .userName)
databricks experiments create-experiment /Users/$DATABRICKS_USERNAME/agent-memory --profile dev
```

## Running Locally

### Start both frontend and backend:

```bash
uv run start-app
```

Opens the chat UI at **http://localhost:8000** (backend) with the Next.js frontend on port 3000.

### Backend only (no UI):

```bash
uv run start-app --no-ui
```

### Advanced server options:

```bash
uv run start-server --reload    # Hot-reload on code changes
uv run start-server --port 8001 # Custom port
uv run start-server --workers 4 # Multiple workers
```

### Query via API:

**Streaming:**

```bash
curl -X POST http://localhost:8000/invocations \
  -H "Content-Type: application/json" \
  -d '{"input": [{"role": "user", "content": "hi"}], "stream": true}'
```

**Non-streaming:**

```bash
curl -X POST http://localhost:8000/invocations \
  -H "Content-Type: application/json" \
  -d '{"input": [{"role": "user", "content": "hi"}]}'
```

## Evaluating the Agent

Update test cases in `agent_server/evaluate_agent.py`, then run:

```bash
uv run agent-evaluate
```

Results are logged to your MLflow experiment for inspection.

## Deploying to Databricks Apps

### Using Databricks Asset Bundles (Recommended)

The `databricks.yml` uses parameterised variables. Supply them at deploy time:

```bash
databricks bundle deploy --profile dev \
  --var="budget_policy_id=<your-budget-policy-id>" \
  --var="pghost=<your-lakebase-host>" \
  --var="pgdatabase=chatbot" \
  --var="pguser=<app-service-principal-client-id>" \
  --var="lakebase_endpoint_path=projects/<name>/branches/<branch>/endpoints/primary" \
  --var="lakebase_project=<project-name>" \
  --var="lakebase_branch=production" \
  --var="mlflow_experiment_id=<experiment-id>"

databricks bundle run agent_openai_agents_sdk --profile dev \
  --var="budget_policy_id=<your-budget-policy-id>" \
  --var="pghost=<your-lakebase-host>" \
  --var="pgdatabase=chatbot" \
  --var="pguser=<app-service-principal-client-id>" \
  --var="lakebase_endpoint_path=projects/<name>/branches/<branch>/endpoints/primary" \
  --var="lakebase_project=<project-name>" \
  --var="lakebase_branch=production" \
  --var="mlflow_experiment_id=<experiment-id>"
```

### Granting Lakebase Permissions

After deploying, the app's service principal needs Lakebase access:

```bash
# Get the SP client ID
databricks apps get <app-name> --output json | jq -r '.service_principal_client_id'

# Grant permissions (run from your user account which owns the Lakebase instance)
uv run python scripts/grant_lakebase_permissions.py <sp-client-id> \
  --memory-type langgraph-short-term \
  --project <lakebase-project> --branch production
```

You may also need to grant database-level `CREATE` and schema privileges directly via SQL for the `agent_memory`, `ai_chatbot`, and `drizzle` schemas.

### Enabling User API Scopes

To enable the `sql` scope for OBO token forwarding (optional, for full OBO Lakebase access):

```bash
databricks apps update <app-name> --json '{"user_api_scopes": ["sql"]}'
```

### View app logs:

```bash
databricks apps logs <app-name> --follow --profile dev
```

### Query the deployed agent:

```bash
# Get an OAuth token
databricks auth token --profile dev

# Send a request
curl -X POST <app-url>.databricksapps.com/invocations \
  -H "Authorization: Bearer <oauth-token>" \
  -H "Content-Type: application/json" \
  -d '{"input": [{"role": "user", "content": "hi"}]}'
```

> **Note:** Databricks Apps only accept OAuth tokens, not Personal Access Tokens (PATs).

## Discovering Available Tools

Find MCP servers, Genie spaces, UC functions, and vector search indexes in your workspace:

```bash
uv run discover-tools
```

## Project Structure

```
agent-memory/
├── agent_server/
│   ├── agent.py              # Agent definition, tools, invoke/stream handlers
│   ├── memory.py             # Key-value memory + pgvector semantic search
│   ├── start_server.py       # FastAPI + MLflow AgentServer entrypoint
│   ├── evaluate_agent.py     # Evaluation with conversation simulation
│   └── utils.py              # Helpers (session, MCP, streaming)
├── scripts/
│   ├── quickstart.py         # One-command setup
│   ├── start_app.py          # Frontend + backend launcher
│   ├── discover_tools.py     # Workspace resource discovery
│   └── grant_lakebase_permissions.py  # Grant SP access to Lakebase
├── e2e-chatbot-app-next/     # Next.js/Express chat frontend with per-user history
│   ├── client/               # React + Vite frontend
│   ├── server/               # Express backend (auth, chat, history routes)
│   └── packages/             # Shared libs (auth, db, ai-sdk-providers, core)
├── databricks.yml            # Asset Bundle config with parameterised variables
├── app.yaml                  # Databricks App runtime config
├── pyproject.toml            # Python dependencies and scripts
├── .env.example              # Template for environment variables
└── .env                      # Local environment config (not committed)
```

## Data Model

### Key-Value Memory (`agent_memory.memories`)

| Column | Type | Description |
|--------|------|-------------|
| `id` | SERIAL | Primary key |
| `user_id` | TEXT | User email |
| `key` | TEXT | Memory key (e.g., `name`, `preferred_language`) |
| `value` | TEXT | Memory value |
| `created_at` | TIMESTAMPTZ | Creation timestamp |
| `updated_at` | TIMESTAMPTZ | Last update timestamp |

Unique constraint on `(user_id, key)` — upserts on conflict.

### Conversation Embeddings (`agent_memory.conversation_embeddings`)

| Column | Type | Description |
|--------|------|-------------|
| `id` | SERIAL | Primary key |
| `user_id` | TEXT | User email (per-user isolation) |
| `chat_id` | TEXT | References the chat this turn belongs to |
| `turn_index` | INTEGER | Position in conversation |
| `content_text` | TEXT | Raw text (User + Assistant combined) |
| `embedding` | vector(1024) | GTE-Large-EN embedding (pgvector) |
| `created_at` | TIMESTAMPTZ | Creation timestamp |

HNSW index on `embedding` with `vector_cosine_ops` for fast similarity search.

### Conversation Summaries (`agent_memory.conversation_summaries`)

| Column | Type | Description |
|--------|------|-------------|
| `id` | SERIAL | Primary key |
| `user_id` | TEXT | User email (per-user isolation) |
| `chat_id` | TEXT | One summary per chat (UNIQUE) |
| `title` | TEXT | Chat title |
| `summary_text` | TEXT | LLM-generated conversation summary |
| `key_topics` | TEXT[] | Extracted topic tags |
| `embedding` | vector(1024) | Embedding of the summary |
| `message_count` | INTEGER | Number of messages summarised |

### Chat History (`ai_chatbot` schema)

| Table | Key Column | Description |
|-------|-----------|-------------|
| `Chat` | `userId` (email) | Conversation metadata, title, visibility |
| `Message` | `chatId` (FK) | Individual messages with role, parts, traceId |
| `User` | `id` | User records |
| `Vote` | `chatId`, `messageId` | User feedback on messages |

## FAQ

**Q: The agent says "No memories stored yet" — is Lakebase connected?**
Ensure `PGHOST`, `PGDATABASE`, and `PGPORT` are set in your `.env`. The schema and table are auto-created on first request via the SP connection.

**Q: How do I change the LLM model?**
Edit the `model` parameter in `create_agent()` in `agent_server/agent.py`.

**Q: Can I add more tools?**
Yes. Define tools with `@function_tool` in `agent.py`, or connect MCP servers for Unity Catalog functions, Vector Search, and Genie Spaces. See `.claude/skills/add-tools/SKILL.md` for guidance.

**Q: A user sees another user's memories or chat history.**
Verify that the `user_id` column in `agent_memory.memories` and `userId` in `ai_chatbot.Chat` both use the user's email address. The frontend extracts this from the `X-Forwarded-Email` header injected by the Databricks Apps reverse proxy.

**Q: The agent greets with a UUID instead of a name.**
This happens when the user identity falls back to the service principal. Ensure the frontend sends `context.user_id` (email) in the request body, and that `_get_first_name()` in `agent.py` returns `None` for non-human identifiers.

**Q: Streaming returns 200 OK but errors in the stream — is that a bug?**
No. The 200 confirms stream setup; errors are delivered within the stream itself.

**Q: I get a 302 when querying the deployed app.**
Use an OAuth token, not a PAT. Run `databricks auth token --profile dev` to get one.

**Q: How do I grant the app SP access to Lakebase?**
Run `scripts/grant_lakebase_permissions.py` with the SP client ID, or grant `CREATE` on the database and `ALL` on the required schemas (`agent_memory`, `ai_chatbot`, `drizzle`, `public`) via SQL.

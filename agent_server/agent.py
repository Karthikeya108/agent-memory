import asyncio
import logging
import re
from datetime import datetime
from typing import AsyncGenerator

import mlflow
from agents import Agent, Runner, function_tool, set_default_openai_api, set_default_openai_client
from agents.tracing import set_trace_processors
from databricks.sdk import WorkspaceClient
from databricks_openai import AsyncDatabricksOpenAI
from databricks_openai.agents import McpServer
from mlflow.genai.agent_server import invoke, stream
from mlflow.types.responses import (
    ResponsesAgentRequest,
    ResponsesAgentResponse,
    ResponsesAgentStreamEvent,
)

from agent_server.memory import (
    UserCredentials,
    delete_memory,
    find_similar_conversations,
    generate_conversation_summary,
    generate_embedding,
    get_chat_history,
    get_memories,
    get_user_credentials,
    initialize_schema,
    initialize_semantic_schema,
    store_conversation_summary,
    store_turn_embedding,
    upsert_memory,
)
from agent_server.utils import (
    build_mcp_url,
    get_session_id,
    get_user_workspace_client,
    process_agent_stream_events,
)

# NOTE: this will work for all databricks models OTHER than GPT-OSS, which uses a slightly different API
set_default_openai_client(AsyncDatabricksOpenAI())
set_default_openai_api("chat_completions")
set_trace_processors([])  # only use mlflow for trace processing
mlflow.openai.autolog()
logging.getLogger("mlflow.utils.autologging_utils").setLevel(logging.ERROR)
logger = logging.getLogger(__name__)


@function_tool
def get_current_time() -> str:
    """Get the current date and time."""
    return datetime.now().isoformat()


async def init_mcp_server(workspace_client: WorkspaceClient):
    return McpServer(
        url=build_mcp_url("/api/2.0/mcp/functions/system/ai", workspace_client=workspace_client),
        name="system.ai UC function MCP server",
        workspace_client=workspace_client,
    )


def _build_memory_context(memories: list[dict]) -> str:
    """Format stored memories as a system-prompt snippet."""
    if not memories:
        return ""
    lines = "\n".join(f"- {m['key']}: {m['value']}" for m in memories)
    return (
        "\n\n## Your memory about this user\n"
        "The following facts were saved from previous conversations. "
        "Use them to personalise your responses:\n"
        f"{lines}"
    )


def _create_memory_tools(creds: UserCredentials, session_id: str | None = None) -> list:
    """
    Return agent tools bound to the current user's credentials and session.

    Creating tools per-request (as closures) lets each call capture the
    correct credentials without relying on global/thread state.
    """

    async def remember(key: str, value: str) -> str:
        """
        Save a fact about the user for future conversations.

        Use a short, descriptive key (e.g. 'preferred_language', 'timezone',
        'name', 'role') and a concise value.
        Call this whenever you learn something useful that should persist
        across sessions.
        """
        await upsert_memory(creds, key, value)
        return f"Memory saved — {key}: {value}"

    async def recall() -> str:
        """
        Retrieve all facts previously saved about the user.

        Call this at the start of a conversation if the system prompt does
        not already contain memory context, or when the user asks what you
        remember about them.
        """
        stored = await get_memories(creds)
        if not stored:
            return "No memories stored yet for this user."
        lines = "\n".join(f"- {m['key']}: {m['value']}" for m in stored)
        return f"Stored memories:\n{lines}"

    async def forget(key: str) -> str:
        """
        Delete a previously saved memory by its key.

        Call this when the user asks you to forget something or when a fact
        is no longer accurate.
        """
        removed = await delete_memory(creds, key)
        return f"Memory {'deleted' if removed else 'not found'}: {key}"

    async def search_past_conversations(query: str) -> str:
        """
        Search past conversations for context related to a topic.

        Call this ONLY when the user explicitly asks about something from a
        previous conversation, wants to continue an earlier topic, or
        references a past discussion.
        Examples: "what did we talk about last time?", "recall our discussion
        about X", "continue where we left off on Y", "remind me what we
        said about Z".
        Do NOT call this automatically — only when the user asks.
        """
        try:
            embedding = await generate_embedding(query)
            similar = await find_similar_conversations(
                creds, embedding, current_chat_id=session_id
            )
        except Exception as exc:
            return f"Could not search past conversations: {exc}"

        if not similar["turns"] and not similar["summaries"]:
            return "No relevant past conversations found for this topic."

        parts = []
        for s in similar["summaries"]:
            title = s.get("title") or "Untitled"
            topics = ", ".join(s.get("key_topics") or [])
            parts.append(
                f"Previous conversation: {title}\n"
                f"Topics: {topics}\n"
                f"{s['summary_text']}"
            )
        for t in similar["turns"]:
            parts.append(f"Relevant exchange:\n{t['content_text'][:500]}")
        return "\n\n---\n\n".join(parts)

    return [
        function_tool(remember),
        function_tool(recall),
        function_tool(forget),
        function_tool(search_past_conversations),
    ]


def _get_first_name(user_id: str) -> str | None:
    """Extract a capitalised first name from an email or username.

    Returns None if the user_id looks like a UUID or service principal ID
    rather than a human-readable name.
    """

    # UUIDs / SP client IDs → no usable name
    if re.fullmatch(r"[0-9a-fA-F-]{36}", user_id):
        return None

    # "karthik.subbarao@databricks.com" → "Karthik"
    local = user_id.split("@")[0]       # strip domain
    first = local.split(".")[0]          # take part before first dot
    first = first.split("_")[0]          # handle underscores too

    # If it still looks like a hex/numeric ID, skip it
    if re.fullmatch(r"[0-9a-fA-F-]+", first):
        return None

    return first.capitalize()


def create_agent(
    creds: UserCredentials,
    memories: list[dict] | None = None,
    session_id: str | None = None,
    mcp_servers: list[McpServer] | None = None,
) -> Agent:
    memory_context = _build_memory_context(memories or [])
    first_name = _get_first_name(creds.user_id)
    if first_name:
        user_context = (
            f"## Current user\n"
            f"The logged-in user's first name is **{first_name}**. Always greet "
            f"them by this name at the start of a conversation.\n\n"
        )
    else:
        user_context = (
            "## Current user\n"
            "The user's name is not yet known. Greet them warmly and "
            "ask for their name so you can remember it for next time.\n\n"
        )
    return Agent(
        name="Agent",
        instructions=(
            "You are a helpful, personalised assistant with persistent memory.\n\n"
            + user_context
            + "## Memory behaviour\n"
            "- When the user shares personal details (name, preferences, role, "
            "interests, etc.), proactively save them with the `remember` tool and "
            "confirm what you saved (e.g. 'Got it — I'll remember that!').\n"
            "- At the start of a conversation, greet the user by their first name "
            "and reference any memories you have about them.\n"
            "- When the user asks what you remember, use the `recall` tool and "
            "share the results.\n"
            "- When the user asks you to forget something, use the `forget` tool.\n"
            "- When the user references a past conversation or asks to continue a "
            "previous topic, use `search_past_conversations` to find relevant context. "
            "Only search when the user explicitly asks — do not search automatically.\n"
            "- Your memory is private to each user — no one else can see it.\n"
            + memory_context
        ),
        model="databricks-gpt-5-2",
        tools=[get_current_time, *_create_memory_tools(creds, session_id)],
        mcp_servers=mcp_servers or [],
    )


def _filter_messages(request: ResponsesAgentRequest) -> list[dict]:
    """Keep only user and assistant text messages.

    The frontend may send back full conversation history that includes
    tool_call / tool result items from previous turns.  The LLM rejects
    these if the preceding tool_calls message is missing, so we strip
    them out.  The agent already has memory context via the system prompt.
    """
    allowed_roles = {"user", "assistant", "system"}
    filtered = []
    for item in request.input:
        d = item.model_dump()
        if d.get("role") in allowed_roles:
            filtered.append(d)
    return filtered


async def _build_messages(
    request: ResponsesAgentRequest, creds: UserCredentials
) -> list[dict]:
    """Build the full message list: chat history + current request messages.

    When a conversation_id is present, loads previous messages from Lakebase
    so the agent can continue the conversation with full context.
    """
    current_messages = _filter_messages(request)

    # Load chat history if this is a continuation of an existing conversation
    conversation_id = get_session_id(request)
    if not conversation_id:
        return current_messages

    history = await get_chat_history(conversation_id, creds)
    if not history:
        return current_messages

    # Deduplicate: if the history already ends with the same user message
    # that starts the current request, don't include it twice.
    if (
        history
        and current_messages
        and history[-1].get("role") == "user"
        and current_messages[0].get("role") == "user"
        and history[-1].get("content") == current_messages[0].get("content")
    ):
        history = history[:-1]

    return history + current_messages


def _extract_last_user_message(messages: list[dict]) -> str | None:
    """Get the text of the last user message."""
    for msg in reversed(messages):
        if msg.get("role") == "user":
            return msg.get("content") or ""
    return None


def _extract_last_assistant_text(new_items) -> str | None:
    """Extract assistant text from Runner result items."""
    for item in reversed(new_items):
        inp = item.to_input_item()
        if inp.get("role") == "assistant":
            for c in inp.get("content", []):
                if isinstance(c, dict) and c.get("type") == "text":
                    return c["text"]
            if isinstance(inp.get("content"), str):
                return inp["content"]
    return None


async def _store_turn_async(
    creds: UserCredentials,
    chat_id: str,
    turn_index: int,
    user_msg: str,
    assistant_msg: str,
) -> None:
    """Background: embed and store a user+assistant turn pair."""
    try:
        content_text = f"User: {user_msg}\nAssistant: {assistant_msg}"
        embedding = await generate_embedding(content_text)
        await store_turn_embedding(creds, chat_id, turn_index, content_text, embedding)
    except Exception as exc:
        logger.warning("[semantic] Failed to store turn embedding: %s", exc)


async def _maybe_update_summary_async(
    creds: UserCredentials,
    chat_id: str,
    messages: list[dict],
) -> None:
    """Background: generate/update conversation summary if enough turns."""
    try:
        if len(messages) < 6:
            return
        summary_text, key_topics = await generate_conversation_summary(messages)
        embedding = await generate_embedding(summary_text)
        await store_conversation_summary(
            creds, chat_id, None, summary_text, key_topics, embedding, len(messages)
        )
    except Exception as exc:
        logger.warning("[semantic] Failed to update summary: %s", exc)


@invoke()
async def invoke_handler(request: ResponsesAgentRequest) -> ResponsesAgentResponse:
    session_id = get_session_id(request)
    if session_id:
        mlflow.update_current_trace(metadata={"mlflow.trace.session": session_id})

    creds = get_user_credentials(request)
    await initialize_schema()
    await initialize_semantic_schema()
    memories = await get_memories(creds)

    agent = create_agent(creds, memories, session_id=session_id)
    messages = await _build_messages(request, creds)
    result = await Runner.run(agent, messages)

    # Background: store turn embedding and maybe update summary
    last_user = _extract_last_user_message(messages)
    last_assistant = _extract_last_assistant_text(result.new_items)
    if session_id and last_user and last_assistant:
        turn_index = len(messages) // 2
        asyncio.create_task(_store_turn_async(creds, session_id, turn_index, last_user, last_assistant))
        asyncio.create_task(_maybe_update_summary_async(creds, session_id, messages))

    return ResponsesAgentResponse(output=[item.to_input_item() for item in result.new_items])


@stream()
async def stream_handler(
    request: ResponsesAgentRequest,
) -> AsyncGenerator[ResponsesAgentStreamEvent, None]:
    session_id = get_session_id(request)
    if session_id:
        mlflow.update_current_trace(metadata={"mlflow.trace.session": session_id})

    creds = get_user_credentials(request)
    await initialize_schema()
    await initialize_semantic_schema()
    memories = await get_memories(creds)

    agent = create_agent(creds, memories, session_id=session_id)
    messages = await _build_messages(request, creds)
    result = Runner.run_streamed(agent, input=messages)

    # Collect assistant text while streaming
    assistant_chunks: list[str] = []
    async for event in process_agent_stream_events(result.stream_events()):
        # Capture text deltas for background storage
        if isinstance(event, dict) and event.get("type") == "response.output_text.delta":
            assistant_chunks.append(event.get("delta", ""))
        yield event

    # Background: store turn embedding after stream completes
    last_user = _extract_last_user_message(messages)
    assistant_text = "".join(assistant_chunks) if assistant_chunks else None
    if session_id and last_user and assistant_text:
        turn_index = len(messages) // 2
        asyncio.create_task(_store_turn_async(creds, session_id, turn_index, last_user, assistant_text))
        asyncio.create_task(_maybe_update_summary_async(creds, session_id, messages))

import logging
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
    get_memories,
    get_user_credentials,
    initialize_schema,
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


def _create_memory_tools(creds: UserCredentials) -> list:
    """
    Return agent tools bound to the current user's OBO credentials.

    Creating tools per-request (as closures) lets each call capture the
    correct credentials without relying on global/thread state.  All
    Lakebase operations use the user's own OAuth token.
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

    return [
        function_tool(remember),
        function_tool(recall),
        function_tool(forget),
    ]


def _get_first_name(user_id: str) -> str | None:
    """Extract a capitalised first name from an email or username.

    Returns None if the user_id looks like a UUID or service principal ID
    rather than a human-readable name.
    """
    import re

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
            f"You are a helpful, personalised assistant with persistent memory.\n\n"
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
            "- Your memory is private to each user — no one else can see it.\n"
            + memory_context
        ),
        model="databricks-gpt-5-2",
        tools=[get_current_time, *_create_memory_tools(creds)],
        mcp_servers=mcp_servers or [],
    )


@invoke()
async def invoke_handler(request: ResponsesAgentRequest) -> ResponsesAgentResponse:
    if session_id := get_session_id(request):
        mlflow.update_current_trace(metadata={"mlflow.trace.session": session_id})

    creds = get_user_credentials(request)
    await initialize_schema()
    memories = await get_memories(creds)

    agent = create_agent(creds, memories)
    messages = [i.model_dump() for i in request.input]
    result = await Runner.run(agent, messages)
    return ResponsesAgentResponse(output=[item.to_input_item() for item in result.new_items])


@stream()
async def stream_handler(
    request: ResponsesAgentRequest,
) -> AsyncGenerator[ResponsesAgentStreamEvent, None]:
    if session_id := get_session_id(request):
        mlflow.update_current_trace(metadata={"mlflow.trace.session": session_id})

    creds = get_user_credentials(request)
    await initialize_schema()
    memories = await get_memories(creds)

    agent = create_agent(creds, memories)
    messages = [i.model_dump() for i in request.input]
    result = Runner.run_streamed(agent, input=messages)

    async for event in process_agent_stream_events(result.stream_events()):
        yield event

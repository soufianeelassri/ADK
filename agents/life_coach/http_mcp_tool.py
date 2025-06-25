import os
import json
import httpx
from google.adk.tools import FunctionTool

MCP_SERVER_URL = os.environ.get("MCP_SERVER_URL")
if not MCP_SERVER_URL:
    raise ValueError("MCP_SERVER_URL environment variable is not set.")

async def _call_mcp_http(user_id: str, message: str):
    """
    Async generator that:
     1) POSTs to /v1/tasks to ensure a task exists.
     2) GETs /v1/tasks/{user_id}/updates?message=… as an SSE stream.
     3) Yields each parsed JSON payload as a Python dict.
    """
    async with httpx.AsyncClient() as client:
        resp = await client.post(
            f"{MCP_SERVER_URL}/v1/tasks",
            json={"user_id": user_id},
            headers={"Content-Type": "application/json"},
            timeout=10.0,
        )
        resp.raise_for_status()

    url = f"{MCP_SERVER_URL}/v1/tasks/{user_id}/updates"
    async with httpx.AsyncClient(timeout=None) as client:
        async with client.stream(
            "GET",
            url,
            params={"message": message},
            headers={"Accept": "text/event-stream"},
        ) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if not line.startswith("data:"):
                    continue
                payload = line.removeprefix("data:").strip()
                if payload == '{"type":"end_of_stream"}':
                    return
                try:
                    yield json.loads(payload)
                except json.JSONDecodeError:
                    continue

MCP_HTTP_TOOL = FunctionTool(
    _call_mcp_http
)

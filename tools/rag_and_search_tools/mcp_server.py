import asyncio
import json
import uvicorn
import os
from dotenv import load_dotenv

from mcp import types as mcp_types
from mcp.server.lowlevel import Server
from mcp.server.sse import SseServerTransport
from starlette.applications import Starlette
from starlette.routing import Route
from starlette.responses import JSONResponse
from starlette.requests import Request

from google.adk.tools.function_tool import FunctionTool
from google.adk.tools.mcp_tool.conversion_utils import adk_to_mcp_tool_type

from knowledge_base import retrieve_from_knowledge_base

load_dotenv()

HOST = os.environ.get("HOST", "0.0.0.0")
PORT = int(os.environ.get("PORT", 8080))

rag_tool = FunctionTool(retrieve_from_knowledge_base)
available_tools = {
    rag_tool.name: rag_tool,
}

app = Server("menovibe-tools-server")
sse = SseServerTransport("/messages/")

@app.list_tools()
async def list_tools() -> list[mcp_types.Tool]:
    mcp_tools = [adk_to_mcp_tool_type(tool) for tool in available_tools.values()]
    print(f"MCP Server: list_tools -> {[tool.name for tool in mcp_tools]}")
    return mcp_tools

@app.call_tool()
async def call_tool(name: str, arguments: dict) -> list[mcp_types.Content]:
    print(f"MCP Server: call_tool -> {name} args={arguments}")
    tool_to_call = available_tools.get(name)
    if not tool_to_call:
        return [mcp_types.TextContent(type="text", text=json.dumps({"error": f"Tool '{name}' not found."}))]
    try:
        adk_response = tool_to_call.run_sync(args=arguments, tool_context=None)
        return [mcp_types.TextContent(type="text", text=json.dumps(adk_response, indent=2))]
    except Exception as e:
        return [mcp_types.TextContent(type="text", text=json.dumps({"error": str(e)}))]

async def root_endpoint(request):
    return JSONResponse({"message": "MCP server is running."})

async def handle_sse(request):
    async with sse.connect_sse(request.scope, request.receive, request._send) as streams:
        await app.run(streams[0], streams[1], app.create_initialization_options())

async def messages_post_handler(request: Request):
    try:
        payload = await request.json()
        messages = payload.get("messages")
        if not messages:
            return JSONResponse({"error": "Missing 'messages' field in JSON body"}, status_code=400)

        first_message = messages[0]
        if first_message["type"] != "text":
            return JSONResponse({"error": "Only 'text' message type is supported"}, status_code=400)

        user_text = first_message["text"]

        results = retrieve_from_knowledge_base(user_text)

        return JSONResponse({"results": results})

    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

starlette_app = Starlette(
    debug=True,
    routes=[
        Route("/", endpoint=root_endpoint),
        Route("/sse", endpoint=handle_sse),
        Route("/messages/", endpoint=messages_post_handler, methods=["POST"]),
    ],
)

if __name__ == "__main__":
    print("Launching MCP Server for MenoVibe Tools...")
    uvicorn.run(starlette_app, host=HOST, port=PORT)
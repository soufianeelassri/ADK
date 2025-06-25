# agents-service/agents/nutrition_expert/a2a_server.py
import os
import uvicorn
from google.adk.cli.fast_api import get_fast_api_app

# The directory where this agent's code resides.
# The server will look for an 'agent.py' file with a 'root_agent' variable here.
AGENT_DIR = os.path.dirname(os.path.abspath(__file__))
# Get the FastAPI app instance using the ADK helper.
# This correctly sets up the Runner, SessionService, and API endpoints.
app = get_fast_api_app(
    agents_dir=AGENT_DIR,
    # Using an in-memory session DB for this microservice.
    # In production, you might point to a shared DB like "sqlite:///./sessions.db"
    session_db_url=None,
)

if __name__ == "__main__":
    # Use the PORT environment variable, defaulting to 8080.
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 8080))
    )
# agents-service/agents/life_coach/a2a_server.py
import os
import uvicorn
from google.adk.cli.fast_api import get_fast_api_app

# The directory where this agent's code resides.
AGENT_DIR = os.path.dirname(os.path.abspath(__file__))

# Get the FastAPI app instance using the ADK helper.
app = get_fast_api_app(
    agents_dir=AGENT_DIR,
    session_db_url=None, # Use in-memory sessions
)

if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 8080))
    )
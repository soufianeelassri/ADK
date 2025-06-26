import os
from google.adk.cli.fast_api import get_fast_api_app
from google.adk.sessions import InMemorySessionService 
import uvicorn

AGENTS_ROOT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'agents')

app = get_fast_api_app(
    agents_dir=AGENTS_ROOT_DIR,
    web=True,
)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
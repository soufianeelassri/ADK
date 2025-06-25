import uvicorn
from common.task_manager import AgentTaskManager
from common.types import AgentCard, Task, Message, TextPart, TaskState, TaskStatus, TaskStatusUpdateEvent

class A2AServer:
    def __init__(self, agent_card: AgentCard, task_manager: AgentTaskManager, host: str, port: int):
        self._agent_card = agent_card
        self._task_manager = task_manager
        self._host = host
        self._port = port
        print(f"A2A Server for '{self._agent_card.name}' initialized.")

    def get_agent_card(self):
        return self._agent_card.model_dump_json()

    async def create_task(self, request):
        body = await request.json()
        user_id = body.get("user_id", "default_user")
        message = body.get("message", "")
        task = self._task_manager.create_task(user_id, message)
        return {"task": task}

    def start(self):
        print(f"Starting A2A Server for {self._agent_card.name} on http://{self._host}:{self._port}")
        print("Note: This is a conceptual server. Actual web server logic would be here.")
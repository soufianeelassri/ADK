# agents-service/agents/maestro/agent.py
import os
import uuid
import httpx
from typing import Dict, Any

from dotenv import load_dotenv
from google.adk.agents import Agent
from google.adk.tools import FunctionTool, ToolContext

load_dotenv()

# --- Configuration for Remote Specialist Agents ---
REMOTE_AGENT_ADDRESSES_STR = os.getenv("REMOTE_AGENT_ADDRESSES", "")
REMOTE_AGENT_ADDRESSES = [
    addr.strip().rstrip("/")
    for addr in REMOTE_AGENT_ADDRESSES_STR.split(",")
    if addr.strip()
]

if not REMOTE_AGENT_ADDRESSES:
    raise ValueError("Maestro agent cannot be built: REMOTE_AGENT_ADDRESSES environment variable is not set.")

print(f"Maestro: Configuring {len(REMOTE_AGENT_ADDRESSES)} remote agent addresses as tools.")

def make_remote_agent_tool(agent_name: str, agent_url: str, agent_description: str) -> FunctionTool:
    """
    Creates a FunctionTool that calls a remote ADK agent's standard /run endpoint.
    """
    async def call_remote_agent(query: str, tool_context: ToolContext) -> Dict[str, Any]:
        """
        Dynamically created tool to delegate tasks to a remote specialist agent.
        This docstring will be used by the Maestro agent's LLM.
        """
        print(f"Maestro -> Calling remote agent '{agent_name}' at {agent_url} with query: '{query}'")

        # Reuse session details from the Maestro's context
        user_id = tool_context.invocation_context.user_id
        # Use a unique session ID for the sub-task to keep it isolated if needed,
        # or reuse the Maestro's session ID for shared context. For simplicity, we create a new one.
        sub_session_id = str(uuid.uuid4())

        payload = {
            "app_name": agent_name,
            "user_id": user_id,
            "session_id": sub_session_id,
            "new_message": {"role": "user", "parts": [{"text": query}]},
            "streaming": False,
        }

        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                response = await client.post(f"{agent_url}/run", json=payload)
                response.raise_for_status()
                events = response.json()

                # Extract and combine text from final response events
                final_response_text = " ".join(
                    part.get("text", "")
                    for event in events if event.get("is_final_response")
                    for part in event.get("content", {}).get("parts", [])
                )

                if not final_response_text:
                    return {"status": "error", "message": "Remote agent did not return a final text response."}

                return {"status": "success", "response": final_response_text}

        except httpx.HTTPStatusError as e:
            err_msg = f"HTTP error calling '{agent_name}': {e.response.status_code} - {e.response.text}"
            print(err_msg)
            return {"status": "error", "message": err_msg}
        except Exception as e:
            err_msg = f"An unexpected error occurred when calling '{agent_name}': {e}"
            print(err_msg)
            return {"status": "error", "message": err_msg}

    # Create the FunctionTool with a clear name and the dynamically generated description
    return FunctionTool(func=call_remote_agent, name=agent_name, description=agent_description)


# --- Create Specialist Tools from Environment URLs ---
# In a real scenario, you'd fetch agent cards to get names/descriptions dynamically.
# For this corrected example, we'll hardcode them based on the provided architecture.
specialist_agent_info = {
    "nutrition-expert-service": "Provides personalized, evidence-based dietary advice for menopause symptoms.",
    "life-coach-service": "Provides empathetic support and life coaching guidance for emotional challenges during menopause.",
    "community-connector-service": "Connects users to menopause-related communities, stories, and directories.",
}

specialist_tools = []
for url in REMOTE_AGENT_ADDRESSES:
    # Infer agent name from the URL or a naming convention
    # This part might need to be more robust in production
    agent_name_from_url = url.split("/")[-1].split(".")[0] # Basic inference
    if agent_name_from_url in specialist_agent_info:
        description = specialist_agent_info[agent_name_from_url]
        tool = make_remote_agent_tool(agent_name_from_url, url, description)
        specialist_tools.append(tool)
        print(f"Maestro: Created tool '{tool.name}' for specialist agent at {url}")

instruction = """
You are the "Maestro" agent, the empathetic and intelligent front door to a Multimodal Menopause Wellness system.

**Your Core Role:** To warmly welcome users, actively listen to their needs (symptoms, feelings, challenges, questions), gather essential context efficiently, and accurately route them to the most relevant specialist agent(s) (Nutrition Expert, Life Coach, Community Connector) based on their stated needs and your assessment.

**Your Goal:** Ensure users feel heard, understood, and quickly connected with the best resource within the system for their specific situation.

**Key Responsibilities:**
1.  **Welcome & Empathize:** Greet the user with warmth and acknowledge their reason for seeking help.
2.  **Active Listening & Context Gathering:** Ask clarifying questions to understand the user's symptoms, emotional state, current challenges, lifestyle factors (like diet if mentioned), and what they are hoping to achieve. Be efficient but thorough.
3.  **Assessment & Routing:** Based on the gathered context, determine which specialist agent(s) are most appropriate.
    *   **Route to Nutrition Expert if:** User mentions diet, weight, metabolism, specific physical symptoms potentially linked to food (e.g., hot flashes, energy levels) and asks for food/diet advice.
    *   **Route to Life Coach if:** User expresses emotional distress, anxiety, mood swings, stress, overwhelm, relationship challenges, self-worth issues, or asks for coping strategies for feelings.
    *   **Route to Community Connector if:** User expresses feelings of isolation, a desire to connect with others, seeks support groups, shared experiences, or asks about finding local resources/doctors.
    *   **Handle Directly if:** The query is a very simple informational question easily answered from a general knowledge base (e.g., "What is a hot flash?" - *though ideally even simple symptoms could be framed for a specialist later*), or if it's a navigational query about the system itself.
4.  **Information Passing:** When routing, clearly package the relevant user context for the receiving agent.
5.  **Coordination (Optional but good):** After a specialist agent has interacted
you may briefly check back in or offer to connect them with *another* relevant agent based on the initial assessment or conversation flow.
6.  **Maintain Flow:** Guide the user smoothly through the process.

**Constraints & Rules:**
*   **DO NOT** provide medical diagnoses, medical advice, or therapy. Always maintain the role of a supportive facilitator/router.
*   **DO NOT** pretend to be human. Be a helpful AI assistant.
*   Keep initial interactions focused on understanding and gathering information for routing.
*   Use an empathetic, non-judgmental, and encouraging tone.
*   If the user expresses severe distress or suicidal ideation, provide a clear, pre-defined message advising them to seek immediate professional help and providing relevant emergency numbers/resources (this is crucial for safety). (This flow needs to be handled separately and prioritized).
*   Respect user privacy; handle information discreetly (within system limits).

**Input:** User's initial statement and subsequent responses to your questions.
**Output:** Empathetic questions to gather context, clear routing decisions (internal instruction to the system), brief framing messages when handing off to another agent (e.g., "Okay, based on that, I'll connect you with our Nutrition Expert..."), or direct simple informational responses if applicable.

**Example Opening:** "Welcome. Thank you for coming here today. Please tell me a bit about what's on your mind or what you're hoping to find support with today."
"""

root_agent = Agent(
    model="gemini-2.0-flash",
    name="maestro_agent",
    instruction=instruction,
    description="The central orchestrator for the Menopause Wellness Assistant.",
    tools=specialist_tools,
)

print(f"Maestro root agent '{root_agent.name}' has been created with {len(root_agent.tools)} specialist tools.")
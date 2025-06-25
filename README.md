# MenoGuide+ Agents Service

## Overview

MenoGuide+ is a modular, AI-powered wellness assistant designed to support users through menopause by providing empathetic guidance, evidence-based nutrition advice, emotional coaching, and community connections. The system is built around a multi-agent architecture, with each agent specializing in a key area of support, and leverages Retrieval-Augmented Generation (RAG) tools for up-to-date, context-aware responses.

## Architecture

- **Maestro Agent**: The central orchestrator and entry point for users. It listens, gathers context, and routes users to the most relevant specialist agent (Nutrition Expert, Life Coach, Community Connector).
- **Specialist Agents**: Each agent is a microservice, containerized and independently deployable:
  - **Nutrition Expert**: Provides evidence-based dietary guidance for menopause symptoms, leveraging a RAG knowledge base.
  - **Life Coach**: Offers emotional support and coaching, helping users navigate psychological challenges.
  - **Community Connector**: Connects users to online communities, local support groups, curated content, and directories of professionals.
- **RAG & Search Tools**: Utilities for knowledge retrieval, document search, and integration with Google Cloud Vertex AI Search.
- **Common Utilities**: Shared code for session management, task handling, and type definitions.

## Agents

### Maestro Agent
- **Role**: Welcomes users, gathers context, and routes them to the appropriate specialist agent.
- **Key Features**:
  - Empathetic, context-aware conversation
  - Intelligent routing based on user needs (diet, emotional, community)
  - Coordination and follow-up
  - Safety checks for distress

### Nutrition Expert
- **Role**: Provides personalized, actionable dietary advice for menopause symptoms.
- **Features**:
  - Uses a RAG knowledge base for evidence-based suggestions
  - Analyzes user diet and symptoms
  - Offers practical tips, recipes, and educational content
  - Strong disclaimers: Not a substitute for medical advice

### Life Coach
- **Role**: Supports users emotionally, helping them build resilience and find coping strategies.
- **Features**:
  - Empathetic, non-judgmental coaching
  - Open-ended questions and validation
  - Suggests evidence-based coping techniques
  - Not a substitute for therapy or clinical care

### Community Connector
- **Role**: Connects users to communities, support groups, curated stories, and professional directories.
- **Features**:
  - Curates online and local resources
  - Shares personal stories and content
  - Provides disclaimers for professional listings
  - Respects user privacy

## RAG & Search Tools

- **Location**: `tools/rag_and_search_tools/`
- **Purpose**: Provides document retrieval and search capabilities, integrating with Google Cloud Vertex AI Search and GCS.
- **Key Files**:
  - `knowledge_base.py`: Retrieval logic for knowledge base queries
  - `mcp_server.py`: API server for tool access (FastAPI/Starlette)
  - `requirements.txt`: Dependencies for RAG tools
  - `Dockerfile`: Containerization for deployment

## Setup & Installation

### Prerequisites
- Python 3.12+
- Google Cloud account with Vertex AI and Discovery Engine enabled
- Docker (for containerized deployment)
- `gcloud` CLI

### Environment Setup
1. **Initialize Google Cloud Project**
   ```sh
   ./init.sh
   ```
   This script stores your Google Cloud Project ID locally.

2. **Set Environment Variables**
   ```sh
   source ./set_env.sh
   ```
   This script authenticates with Google Cloud and exports required environment variables.

### Install Dependencies
Each agent and tool has its own `requirements.txt`. For example, to install dependencies for the Nutrition Expert:
```sh
pip install -r agents/nutrition_expert/requirements.txt
```
Repeat for each agent and tool as needed.

## Deployment

### Docker
Each agent and the RAG tools can be built and run as Docker containers. Example for the Nutrition Expert:
```sh
docker build -t nutrition-expert ./agents/nutrition_expert

docker run -p 8080:8080 --env-file .env nutrition-expert
```
Repeat for each agent and the RAG tools.

### Vertex AI Agent Engine
The Maestro agent can be deployed to Google Vertex AI Agent Engine using the provided deployment script:
```sh
python agents/app/agent_engine_app.py --project-id <YOUR_PROJECT_ID> --location <REGION> --agent-name maestro-wellness-agent --remote-agents <SPECIALIST_AGENT_URLS>
```

## Directory Structure

- `agents/` - All agent microservices and shared code
  - `maestro/` - Maestro agent (orchestrator)
  - `nutrition_expert/` - Nutrition Expert agent
  - `life_coach/` - Life Coach agent
  - `community_connector/` - Community Connector agent
  - `common/` - Shared utilities and types
  - `app/` - Deployment and orchestration scripts
- `tools/` - RAG and search tools
- `init.sh`, `set_env.sh` - Setup scripts

## Contact
For questions or contributions, please contact the project maintainer.

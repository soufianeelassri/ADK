FROM python:3.12-slim

WORKDIR /app

ENV PYTHONPATH=/app

COPY ./agents/community_connector/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY ./agents/common /app/agents/common
COPY ./agents/community_connector /app/agents/community_connector

EXPOSE 8080

CMD ["uvicorn", "agents.community_connector.a2a_server:app", "--host", "0.0.0.0", "--port", "8080"]
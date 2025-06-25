FROM python:3.12-slim

WORKDIR /app

ENV PYTHONPATH=/app

COPY ./agents/nutrition_expert/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY ./agents/common /app/agents/common
COPY ./agents/nutrition_expert /app/agents/nutrition_expert

EXPOSE 8080

CMD ["uvicorn", "agents.nutrition_expert.a2a_server:app", "--host", "0.0.0.0", "--port", "8080"]
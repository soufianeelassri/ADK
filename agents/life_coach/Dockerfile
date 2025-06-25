FROM python:3.12-slim

WORKDIR /app

ENV PYTHONPATH=/app

COPY ./agents/life_coach/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY ./agents/common /app/agents/common
COPY ./agents/life_coach /app/agents/life_coach

EXPOSE 8080

CMD ["uvicorn", "agents.life_coach.a2a_server:app", "--host", "0.0.0.0", "--port", "8080"]
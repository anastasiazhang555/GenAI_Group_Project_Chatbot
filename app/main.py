# app/main.py
"""
FastAPI entrypoint for the Empathetic Therapist Chatbot.

Exposes:
- GET /health   : simple health check
- POST /chat    : main chat endpoint, returns reply + emotion + confidence

This file is mainly for:
- Docker deployment (as required by the course rubric)
- Programmatic access (curl, Postman, etc.)

It reuses the core logic from app.chatbot (LSTM emotion + OpenAI).
"""

from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.responses import HTMLResponse
from app.chatbot import generate_response_with_metadata


# -----------------------------
# 1. Request / Response models
# -----------------------------

class ChatRequest(BaseModel):
    message: str


class ChatResponse(BaseModel):
    reply: str
    emotion: str
    confidence: float


# -----------------------------
# 2. FastAPI app
# -----------------------------

app = FastAPI(
    title="Empathetic Therapist Chatbot API",
    description=(
        "FastAPI wrapper around the LSTM emotion classifier "
        "and the OpenAI-based therapist-style chatbot."
    ),
    version="1.0.0",
)

@app.get("/", response_class=HTMLResponse)
async def root():
    """
    Simple landing page for the instructor when they open http://localhost:8000
    """
    return """
    <html>
      <head>
        <title>Empathetic Therapist Chatbot API</title>
      </head>
      <body style="font-family: -apple-system, system-ui, sans-serif; max-width: 800px; margin: 2rem auto;">
        <h1>🧠 Empathetic Therapist Chatbot API</h1>
        <p>The FastAPI backend is running correctly.</p>
        <p>Useful links:</p>
        <ul>
          <li><a href="/docs">Interactive API documentation (/docs)</a></li>
          <li><a href="/health">Health check endpoint (/health)</a></li>
        </ul>
        <p>You can send POST requests to <code>/chat</code> with JSON like:</p>
        <pre>{
  "message": "I feel a bit anxious about my exams."
}</pre>
      </body>
    </html>
    """

@app.get("/health")
async def health():
    """
    Simple healthcheck endpoint for the instructor or Docker tests.
    """
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Main chat endpoint.

    Example request JSON:
    {
      "message": "I feel very stressed about my future."
    }

    Example response JSON:
    {
      "reply": "...",
      "emotion": "sadness",
      "confidence": 0.93
    }
    """
    reply_text, emotion, confidence = generate_response_with_metadata(request.message)
    return ChatResponse(
        reply=reply_text,
        emotion=emotion,
        confidence=confidence,
    )


# -----------------------------
# 3. Local dev entrypoint
# -----------------------------
# So you can run:  python -m app.main
# (For Docker, we'll use `uvicorn app.main:app` instead.)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )
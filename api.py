"""FastAPI REST API Service for the Personal Python QA Chatbot.

Provides programmatic endpoints for question answering, category browsing,
and random topic sampling.
"""

from typing import Any, Dict, List, Optional
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.bot import PythonQABot, BotResponse
from src.knowledge_base import KBEntry

app = FastAPI(
    title="Python QA Chatbot API",
    description="RESTful API for querying Python programming knowledge base via natural language.",
    version="1.0.0"
)

# Enable CORS for browser integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize bot singleton
bot = PythonQABot()


# Schemas
class AskRequest(BaseModel):
    query: str = Field(..., description="The user's question or statement in natural language", example="What is a decorator?")


class KBEntryResponse(BaseModel):
    id: str
    category: str
    difficulty: str
    question: str
    patterns: List[str]
    tags: List[str]
    summary: str
    answer: str
    code_example: str


class AskResponse(BaseModel):
    query: str
    intent: str
    found: bool
    confidence: float
    text: str
    entry: Optional[KBEntryResponse] = None
    suggestions: List[str] = Field(default_factory=list)
    related_questions: List[str] = Field(default_factory=list)


class HealthResponse(BaseModel):
    status: str
    total_entries: int
    categories: List[str]
    version: str


@app.get("/health", response_model=HealthResponse, tags=["Health"])
def health_check():
    """Returns system status and knowledge base summary."""
    return {
        "status": "healthy",
        "total_entries": bot.kb.count(),
        "categories": bot.kb.get_categories(),
        "version": "1.0.0"
    }


@app.post("/ask", response_model=AskResponse, tags=["Q&A"])
def ask_question(request: AskRequest):
    """Answers a Python question asked in natural language."""
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    res: BotResponse = bot.ask(request.query)
    return res.to_dict()


@app.get("/topics", response_model=List[str], tags=["Knowledge Base"])
def get_topics():
    """Lists all available topic categories."""
    return bot.kb.get_categories()


@app.get("/questions", response_model=List[KBEntryResponse], tags=["Knowledge Base"])
def get_questions(category: Optional[str] = Query(None, description="Optional category filter")):
    """Retrieves all questions, optionally filtered by category."""
    if category:
        entries = bot.kb.get_by_category(category)
    else:
        entries = bot.kb.get_all()
    return [e.to_dict() for e in entries]


@app.get("/random", response_model=AskResponse, tags=["Q&A"])
def get_random_question():
    """Returns a random question from the knowledge base."""
    res = bot.ask("/random")
    return res.to_dict()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="127.0.0.1", port=8000, reload=True)


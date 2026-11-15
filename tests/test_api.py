"""Unit tests for FastAPI endpoints."""

import pytest
from fastapi import HTTPException
from api import health_check, ask_question, get_topics, get_questions, get_random_question, AskRequest


def test_api_health():
    data = health_check()
    assert data["status"] == "healthy"
    assert data["total_entries"] > 0
    assert len(data["categories"]) > 0


def test_api_topics():
    topics = get_topics()
    assert isinstance(topics, list)
    assert len(topics) > 0


def test_api_ask_valid_query():
    req = AskRequest(query="What is the difference between list and tuple?")
    data = ask_question(req)
    assert data["found"] is True
    assert data["entry"] is not None
    assert "list" in data["entry"]["question"].lower()
    assert data["confidence"] > 0


def test_api_ask_conversational():
    req = AskRequest(query="hello")
    data = ask_question(req)
    assert data["intent"] == "greeting"
    assert data["found"] is True


def test_api_ask_empty_query():
    req = AskRequest(query="   ")
    with pytest.raises(HTTPException) as exc_info:
        ask_question(req)
    assert exc_info.value.status_code == 400


def test_api_questions_list():
    entries = get_questions(category=None)
    assert isinstance(entries, list)
    assert len(entries) > 0

    cat_entries = get_questions(category="Data Structures")
    assert len(cat_entries) > 0
    for e in cat_entries:
        assert e["category"] == "Data Structures"


def test_api_random():
    data = get_random_question()
    assert data["intent"] == "random"
    assert data["entry"] is not None

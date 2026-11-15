"""Integration tests for the PythonQABot controller."""

import pytest
from src.bot import PythonQABot


@pytest.fixture
def bot():
    return PythonQABot()


def test_bot_greetings(bot):
    res = bot.ask("Hello there!")
    assert res.intent == "greeting"
    assert res.found is True
    assert "Hello" in res.text or "Hi" in res.text or "Hey" in res.text


def test_bot_farewell(bot):
    res = bot.ask("bye")
    assert res.intent == "farewell"
    assert res.found is True
    assert "Goodbye" in res.text or "See you" in res.text or "Bye" in res.text


def test_bot_about(bot):
    res = bot.ask("who are you?")
    assert res.intent == "about"
    assert res.found is True
    assert "Personal Python Question-Answering Chatbot" in res.text


def test_bot_help_and_topics(bot):
    res_help = bot.ask("/help")
    assert res_help.intent == "help"
    assert "Quick Guide" in res_help.text

    res_topics = bot.ask("/topics")
    assert res_topics.intent == "topics"
    assert "Topic Categories" in res_topics.text


def test_bot_random_question(bot):
    res = bot.ask("/random")
    assert res.intent == "random"
    assert res.entry is not None
    assert res.found is True


def test_bot_confident_python_question(bot):
    res = bot.ask("What is a decorator?")
    assert res.intent == "qa"
    assert res.found is True
    assert res.entry is not None
    assert res.entry.id == "decorators"
    assert "decorator" in res.text.lower()
    assert res.confidence >= 0.30


def test_bot_low_confidence_fallback(bot):
    res = bot.ask("How to build an airplane rocket in aerospace engineering?")
    assert res.intent == "qa"
    assert res.found is False
    assert len(res.suggestions) > 0
    assert "Did you mean one of these Python topics?" in res.text


def test_bot_history_management(bot):
    bot.clear_history()
    assert len(bot.get_history()) == 0

    bot.ask("Hello")
    bot.ask("What is a tuple?")
    assert len(bot.get_history()) == 2

    bot.clear_history()
    assert len(bot.get_history()) == 0


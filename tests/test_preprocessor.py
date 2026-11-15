"""Unit tests for the Preprocessor module."""

import pytest
from src.preprocessor import normalize_text, tokenize, get_clean_string


def test_normalize_text_basic():
    text = "What is Python?"
    norm = normalize_text(text)
    assert norm == "what is python"


def test_normalize_synonyms():
    # Plural forms to singular
    assert "list" in normalize_text("Difference between lists and tuples")
    assert "dict" in normalize_text("How do dictionaries work in Python?")
    assert "decorator" in normalize_text("Explain decorators")


def test_preserve_python_operators():
    text = "What does := do in Python?"
    norm = normalize_text(text)
    assert "walrus" in norm

    eq_text = "Difference between is and =="
    eq_norm = normalize_text(eq_text)
    assert "equality" in eq_norm


def test_preserve_whitelisted_python_keywords():
    tokens = tokenize("if not x in list and y is None:")
    assert "not" in tokens
    assert "in" in tokens
    assert "is" in tokens
    assert "and" in tokens


def test_get_clean_string():
    cleaned = get_clean_string("Can you please explain what a decorator in Python is?")
    assert "decorator" in cleaned
    assert "python" in cleaned
    # Filler words stripped
    assert "please" not in cleaned.split()


"""Unit tests for the Natural Language Q&A Matching Engine."""

from pathlib import Path
import pytest
from src.knowledge_base import KnowledgeBase
from src.engine import QAMatchingEngine


@pytest.fixture
def engine():
    kb_path = Path(__file__).resolve().parent.parent / "data" / "python_kb.json"
    kb = KnowledgeBase(str(kb_path))
    return QAMatchingEngine(kb)


def test_engine_exact_match(engine):
    results = engine.search("What is Python and what are its key features?")
    assert len(results) > 0
    top = results[0]
    assert top.entry.id == "what_is_python"
    assert top.score >= 0.90


def test_engine_pattern_match(engine):
    # Pattern test: "List vs tuple"
    results = engine.search("List vs tuple")
    assert len(results) > 0
    assert results[0].entry.id == "list_vs_tuple"


def test_engine_natural_language_paraphrasing(engine):
    queries_and_expected = [
        ("tell me about decorators in python", "decorators"),
        ("what is global interpreter lock", "gil_in_python"),
        ("how does yield work in generator function", "generators_and_yield"),
        ("why should i not use a mutable list as default argument", "mutable_default_arguments"),
        ("explain the walrus operator", "walrus_operator"),
        ("difference between shallow copy and deep copy", "shallow_vs_deep_copy"),
        ("how does inheritance and mro work", "inheritance_and_mro"),
    ]

    for query, expected_id in queries_and_expected:
        results = engine.search(query, top_k=3)
        assert len(results) > 0, f"No results for query: {query}"
        top_ids = [r.entry.id for r in results]
        assert expected_id in top_ids, f"Expected {expected_id} in {top_ids} for query '{query}'"


def test_engine_gibberish_handling(engine):
    results = engine.search("xyzqwerty foobarbaz nonexistingtopic12345")
    # Low confidence or empty
    if results:
        assert results[0].score < 0.35


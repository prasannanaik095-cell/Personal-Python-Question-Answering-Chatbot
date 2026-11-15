"""Unit tests for the Knowledge Base module."""

from pathlib import Path
import pytest
from src.knowledge_base import KnowledgeBase, KBEntry


@pytest.fixture
def kb():
    kb_path = Path(__file__).resolve().parent.parent / "data" / "python_kb.json"
    return KnowledgeBase(str(kb_path))


def test_kb_load_entries(kb):
    assert kb.count() >= 20
    entries = kb.get_all()
    assert len(entries) == kb.count()


def test_kb_entry_validity(kb):
    ids = set()
    valid_difficulties = {"Beginner", "Intermediate", "Advanced"}

    for entry in kb.get_all():
        assert entry.id, "Entry must have an ID"
        assert entry.id not in ids, f"Duplicate ID found: {entry.id}"
        ids.add(entry.id)

        assert entry.question, f"Entry {entry.id} missing question"
        assert entry.category, f"Entry {entry.id} missing category"
        assert entry.difficulty in valid_difficulties, f"Invalid difficulty: {entry.difficulty}"
        assert entry.answer, f"Entry {entry.id} missing answer"
        assert len(entry.patterns) > 0, f"Entry {entry.id} should have alternate patterns"
        assert len(entry.tags) > 0, f"Entry {entry.id} should have tags"


def test_kb_categories(kb):
    cats = kb.get_categories()
    assert len(cats) >= 5
    assert "Basics & Syntax" in cats or "Data Structures" in cats

    # Test filtering by category
    cat_entries = kb.get_by_category("Data Structures")
    assert len(cat_entries) > 0
    for e in cat_entries:
        assert e.category == "Data Structures"


def test_kb_lookup_by_id(kb):
    entry = kb.get_by_id("list_vs_tuple")
    assert entry is not None
    assert "list" in entry.question.lower()
    assert "tuple" in entry.question.lower()


def test_kb_random_entry(kb):
    entry = kb.get_random()
    assert isinstance(entry, KBEntry)
    assert entry.id


def test_kb_search_by_tag(kb):
    results = kb.search_by_tag("decorator")
    assert len(results) >= 1
    assert any("decorator" in e.id for e in results)


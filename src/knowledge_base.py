"""Knowledge Base loader and management module for Python QA Chatbot."""

from dataclasses import dataclass, field
import json
from pathlib import Path
import random
from typing import Any, Dict, List, Optional


@dataclass
class KBEntry:
    """Represents a single Python Q&A knowledge base entry."""

    id: str
    category: str
    difficulty: str
    question: str
    patterns: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    summary: str = ""
    answer: str = ""
    code_example: str = ""

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "KBEntry":
        """Instantiate a KBEntry from a JSON-derived dictionary."""
        return cls(
            id=data.get("id", ""),
            category=data.get("category", "General"),
            difficulty=data.get("difficulty", "Beginner"),
            question=data.get("question", ""),
            patterns=data.get("patterns", []),
            tags=data.get("tags", []),
            summary=data.get("summary", ""),
            answer=data.get("answer", ""),
            code_example=data.get("code_example", "")
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert entry into serializable dictionary."""
        return {
            "id": self.id,
            "category": self.category,
            "difficulty": self.difficulty,
            "question": self.question,
            "patterns": self.patterns,
            "tags": self.tags,
            "summary": self.summary,
            "answer": self.answer,
            "code_example": self.code_example
        }

    def get_searchable_corpus(self) -> str:
        """Constructs an enriched, weighted text representation for vector indexing.

        Repeats high-signal components (question title, patterns, tags) to give them
        higher natural term frequency.
        """
        parts: List[str] = []

        # Question title repeated 3x for weight
        parts.extend([self.question] * 3)

        # Patterns repeated 2x
        for pattern in self.patterns:
            parts.extend([pattern] * 2)

        # Tags repeated 4x for strong keyword matching
        parts.extend([" ".join(self.tags)] * 4)

        # Summary and category
        if self.summary:
            parts.append(self.summary)
        parts.append(self.category)

        # Answer content
        if self.answer:
            parts.append(self.answer)

        return " ".join(parts)


class KnowledgeBase:
    """In-memory repository of Python questions and answers loaded from JSON."""

    def __init__(self, file_path: Optional[str] = None):
        self.entries: List[KBEntry] = []
        self._entries_by_id: Dict[str, KBEntry] = {}
        self._categories: List[str] = []
        self.file_path = file_path

        if file_path:
            self.load(file_path)

    def load(self, file_path: str) -> None:
        """Loads and parses the knowledge base JSON file."""
        path = Path(file_path)
        if not path.is_file():
            raise FileNotFoundError(f"Knowledge base file not found at: {file_path}")

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if not isinstance(data, list):
            raise ValueError(f"Knowledge base must be a JSON array of objects, got {type(data)}")

        self.entries = [KBEntry.from_dict(item) for item in data]
        self._entries_by_id = {e.id: e for e in self.entries}

        # Collect unique categories preserving order
        seen_cats = set()
        cats = []
        for e in self.entries:
            if e.category not in seen_cats:
                seen_cats.add(e.category)
                cats.append(e.category)
        self._categories = sorted(cats)

    def get_by_id(self, entry_id: str) -> Optional[KBEntry]:
        """Fetch an entry by its unique ID."""
        return self._entries_by_id.get(entry_id)

    def get_all(self) -> List[KBEntry]:
        """Return all entries."""
        return list(self.entries)

    def count(self) -> int:
        """Return total number of entries."""
        return len(self.entries)

    def get_categories(self) -> List[str]:
        """Return list of distinct topic categories."""
        return list(self._categories)

    def get_by_category(self, category: str) -> List[KBEntry]:
        """Return all entries under a given category."""
        cat_lower = category.lower()
        return [e for e in self.entries if e.category.lower() == cat_lower]

    def get_random(self) -> Optional[KBEntry]:
        """Return a random entry from the knowledge base."""
        if not self.entries:
            return None
        return random.choice(self.entries)

    def search_by_tag(self, tag: str) -> List[KBEntry]:
        """Find entries containing a specific tag."""
        tag_lower = tag.lower()
        return [e for e in self.entries if any(tag_lower in t.lower() for t in e.tags)]


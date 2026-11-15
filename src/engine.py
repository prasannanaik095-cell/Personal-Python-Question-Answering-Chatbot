"""Hybrid Natural Language Question Matching Engine for Python QA Chatbot.

Combines TF-IDF vectorization with Cosine Similarity, Exact/Pattern
matching, and Domain Keyword boosting to achieve accurate question retrieval.
"""

from dataclasses import dataclass
from typing import List, Optional, Set, Tuple
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.knowledge_base import KnowledgeBase, KBEntry
from src.preprocessor import get_clean_string, normalize_text, tokenize


@dataclass
class MatchResult:
    """Represents a matched question with its relevance score."""
    entry: KBEntry
    score: float
    match_type: str  # "exact", "high_confidence", "probable", "low_confidence"


class QAMatchingEngine:
    """Hybrid Question-Answering search engine."""

    def __init__(self, kb: KnowledgeBase):
        self.kb = kb
        self.vectorizer: Optional[TfidfVectorizer] = None
        self.tfidf_matrix = None
        self._corpus: List[str] = []
        self._index_to_entry: List[KBEntry] = []
        self._built = False

        if kb.count() > 0:
            self.build_index()

    def build_index(self) -> None:
        """Builds TF-IDF matrices and preprocessed search indexes."""
        entries = self.kb.get_all()
        if not entries:
            return

        self._index_to_entry = entries
        self._corpus = []

        for entry in entries:
            # Preprocess the rich searchable text for each entry
            enriched_text = entry.get_searchable_corpus()
            cleaned = get_clean_string(enriched_text)
            self._corpus.append(cleaned)

        # Build TF-IDF vectorizer with unigram and bigram recognition
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            sublinear_tf=True,
            norm="l2",
            lowercase=True
        )
        self.tfidf_matrix = self.vectorizer.fit_transform(self._corpus)
        self._built = True

    def _check_exact_or_pattern_match(self, normalized_query: str) -> Optional[Tuple[KBEntry, float]]:
        """Checks for direct exact match or high pattern similarity."""
        query_tokens = set(tokenize(normalized_query))
        if not query_tokens:
            return None

        best_entry: Optional[KBEntry] = None
        best_ratio = 0.0

        for entry in self._index_to_entry:
            # Check normalized question
            norm_q = normalize_text(entry.question)
            if normalized_query == norm_q:
                return (entry, 1.0)

            # Check all alternate patterns
            for pattern in entry.patterns:
                norm_pat = normalize_text(pattern)
                if normalized_query == norm_pat:
                    return (entry, 0.99)

                # Token set Jaccard similarity for short queries
                pat_tokens = set(tokenize(norm_pat))
                if not pat_tokens:
                    continue
                intersection = len(query_tokens & pat_tokens)
                union = len(query_tokens | pat_tokens)
                jaccard = intersection / union if union > 0 else 0.0

                if jaccard > best_ratio:
                    best_ratio = jaccard
                    best_entry = entry

        # If high Jaccard overlap on specific phrasing
        if best_ratio >= 0.85 and best_entry:
            return (best_entry, min(0.98, best_ratio + 0.1))

        return None

    def _calculate_tag_boost(self, query_tokens: Set[str], entry: KBEntry) -> float:
        """Calculates a bonus score if user query matches technical tags directly."""
        if not query_tokens or not entry.tags:
            return 0.0

        entry_tags_set = {t.lower().replace("-", " ") for t in entry.tags}
        # Count overlapping tokens
        matches = 0
        for tag in entry_tags_set:
            tag_words = set(tag.split())
            if tag_words and tag_words.issubset(query_tokens):
                matches += 1

        if matches >= 2:
            return 0.20
        elif matches == 1:
            return 0.10
        return 0.0

    def search(self, query: str, top_k: int = 3) -> List[MatchResult]:
        """Searches the knowledge base for entries best matching the query."""
        if not self._built or not query.strip():
            return []

        norm_query = normalize_text(query)
        clean_query = get_clean_string(query)
        query_tokens = set(tokenize(norm_query))

        if not clean_query:
            return []

        # 1. Direct / Pattern Exact Match check
        exact_match = self._check_exact_or_pattern_match(norm_query)
        exact_entry_id = None
        if exact_match and exact_match[1] >= 0.95:
            exact_entry, exact_score = exact_match
            exact_entry_id = exact_entry.id
            primary_result = MatchResult(
                entry=exact_entry,
                score=round(exact_score, 4),
                match_type="exact"
            )
            # Find related secondary results using TF-IDF
            results = [primary_result]
        else:
            results = []

        # 2. TF-IDF + Cosine Similarity
        query_vec = self.vectorizer.transform([clean_query])
        similarities = cosine_similarity(query_vec, self.tfidf_matrix).flatten()

        # Combine TF-IDF scores with tag boost
        combined_scores: List[Tuple[int, float]] = []
        for idx, sim in enumerate(similarities):
            entry = self._index_to_entry[idx]
            tag_boost = self._calculate_tag_boost(query_tokens, entry)
            final_score = float(sim) + tag_boost

            # Scale to max 1.0
            final_score = min(1.0, max(0.0, final_score))
            combined_scores.append((idx, final_score))

        # Sort by score descending
        combined_scores.sort(key=lambda x: x[1], reverse=True)

        # Build list of top results
        for idx, score in combined_scores:
            entry = self._index_to_entry[idx]
            if exact_entry_id and entry.id == exact_entry_id:
                continue

            if score >= 0.60:
                m_type = "high_confidence"
            elif score >= 0.35:
                m_type = "probable"
            else:
                m_type = "low_confidence"

            results.append(MatchResult(
                entry=entry,
                score=round(score, 4),
                match_type=m_type
            ))
            if len(results) >= top_k:
                break

        return results


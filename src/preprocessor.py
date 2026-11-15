"""Text Preprocessing and Normalization Module for Python QA Chatbot.

Provides specialized tokenization and normalization that preserves
Python-specific syntactic keywords, dunder methods, operators, and decorators.
"""

import re
from typing import List, Set

# Crucial Python keywords and tokens that must NOT be stripped or treated as stop words
PYTHON_KEYWORDS_WHITELIST: Set[str] = {
    "is", "not", "in", "and", "or", "for", "while", "with", "if", "else",
    "elif", "def", "class", "return", "yield", "lambda", "try", "except",
    "finally", "raise", "from", "import", "as", "global", "nonlocal", "pass",
    "break", "continue", "self", "cls", "async", "await", "none", "true", "false"
}

# Standard English stop words that carry little conceptual weight in technical queries
STANDARD_STOP_WORDS: Set[str] = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an",
    "any", "are", "aren't", "as", "at", "be", "because", "been", "before",
    "being", "below", "between", "both", "but", "by", "can", "cannot", "could",
    "did", "do", "does", "doing", "down", "during", "each", "few", "further",
    "had", "has", "have", "having", "he", "her", "here", "hers", "herself",
    "him", "himself", "his", "how", "i", "into", "its", "itself", "me",
    "more", "most", "my", "myself", "no", "nor", "off", "on", "once", "only",
    "other", "ought", "our", "ours", "ourselves", "out", "over", "own",
    "same", "she", "should", "so", "some", "such", "than", "that", "the",
    "their", "theirs", "them", "themselves", "then", "there", "these", "they",
    "this", "those", "through", "to", "too", "under", "until", "up", "very",
    "was", "we", "were", "what", "when", "where", "which", "while", "who",
    "whom", "why", "would", "you", "your", "yours", "yourself", "yourselves",
    # Question filler words:
    "explain", "tell", "show", "describe", "please", "mean", "work", "works"
}

# Synonyms and common morphological forms in Python questions
CANONICAL_SYNONYMS = {
    "lists": "list",
    "tuples": "tuple",
    "dictionaries": "dict",
    "dictionary": "dict",
    "dicts": "dict",
    "functions": "function",
    "funcs": "function",
    "classes": "class",
    "objects": "object",
    "methods": "method",
    "decorators": "decorator",
    "generators": "generator",
    "iterators": "iterator",
    "exceptions": "exception",
    "errors": "exception",
    "modules": "module",
    "packages": "package",
    "comprehensions": "comprehension",
    "fstring": "fstrings",
    "f-string": "fstrings",
    "f-strings": "fstrings",
    "deepcopy": "deep copy",
    "shallowcopy": "shallow copy",
    "pep-8": "pep8",
    "pep_8": "pep8",
    ":=": "walrus",
    "==": "equality",
    "!=": "inequality",
    "**kwargs": "kwargs",
    "*args": "args",
    "__init__": "init constructor",
    "__str__": "str dunder",
    "__repr__": "repr dunder",
    "__enter__": "enter context",
    "__exit__": "exit context",
}


def normalize_text(text: str) -> str:
    """Normalize input text by lowercasing, handling punctuation,

    and standardizing Python terminology while preserving programming tokens.
    """
    if not text:
        return ""

    text = text.strip()

    # Protect operators and special terms before punctuation stripping
    if ":=" in text:
        text = text.replace(":=", " walrus ")
    if "==" in text:
        text = text.replace("==", " equality ")
    if "!=" in text:
        text = text.replace("!=", " inequality ")

    # Normalize case
    text = text.lower()

    # Apply synonym replacements
    for pattern, replacement in CANONICAL_SYNONYMS.items():
        # Match whole words or exact terms
        if pattern in text:
            text = re.sub(rf"\b{re.escape(pattern)}\b", replacement, text)

    # Clean characters but keep alphanumeric, underscores (for dunders), and hyphens
    text = re.sub(r"[^\w\s_]", " ", text)

    # Collapse multiple spaces
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize(text: str, remove_stopwords: bool = True) -> List[str]:
    """Tokenize text into cleaned words, optionally filtering non-whitelisted stopwords."""
    normalized = normalize_text(text)
    raw_tokens = normalized.split()

    tokens: List[str] = []
    for token in raw_tokens:
        token = token.strip("_") if token.startswith("__") and token.endswith("__") else token
        if not token:
            continue
        if remove_stopwords:
            if token in PYTHON_KEYWORDS_WHITELIST or token not in STANDARD_STOP_WORDS:
                tokens.append(token)
        else:
            tokens.append(token)

    return tokens


def get_clean_string(text: str) -> str:
    """Convenience helper returning space-separated cleaned tokens."""
    return " ".join(tokenize(text, remove_stopwords=True))


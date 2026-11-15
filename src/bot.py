"""Chatbot Core Controller Module.

Orchestrates intent parsing, conversational dialogues, knowledge base matching,
confidence thresholding, and response synthesis.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional
import random

from src.knowledge_base import KnowledgeBase, KBEntry
from src.engine import QAMatchingEngine, MatchResult
from src.preprocessor import normalize_text


@dataclass
class BotResponse:
    """Standardized response structure returned by the chatbot."""
    query: str
    intent: str  # "qa", "greeting", "farewell", "about", "help", "topics", "random", "thanks", "unknown"
    found: bool
    confidence: float = 0.0
    text: str = ""
    entry: Optional[KBEntry] = None
    suggestions: List[str] = field(default_factory=list)
    related_questions: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert response to dictionary for JSON APIs."""
        return {
            "query": self.query,
            "intent": self.intent,
            "found": self.found,
            "confidence": self.confidence,
            "text": self.text,
            "entry": self.entry.to_dict() if self.entry else None,
            "suggestions": self.suggestions,
            "related_questions": self.related_questions
        }


# Conversational Intent Triggers
GREETING_WORDS = {
    "hello", "hi", "hey", "howdy", "hola", "greetings",
    "good morning", "good afternoon", "good evening", "sup", "yo"
}

FAREWELL_WORDS = {
    "bye", "goodbye", "see you", "cya", "farewell", "exit", "quit"
}

ABOUT_PHRASES = {
    "who are you", "what are you", "what is your name", "what can you do",
    "introduce yourself", "about you", "tell me about yourself", "your purpose"
}

HELP_WORDS = {
    "help", "/help", "commands", "how to use", "options", "usage", "info"
}

TOPIC_WORDS = {
    "topics", "categories", "/topics", "what topics", "list topics",
    "list categories", "modules", "what do you know"
}

RANDOM_WORDS = {
    "random", "trivia", "quiz", "/random", "surprise me", "random question", "test me"
}

THANKS_WORDS = {
    "thanks", "thank you", "thx", "awesome", "great", "helpful", "perfect", "cool"
}


class PythonQABot:
    """Personal Python Question-Answering Chatbot."""

    DEFAULT_KB_PATH = Path(__file__).resolve().parent.parent / "data" / "python_kb.json"

    def __init__(self, kb_path: Optional[str] = None, confidence_threshold: float = 0.30):
        path_to_load = kb_path if kb_path else str(self.DEFAULT_KB_PATH)
        self.kb = KnowledgeBase(path_to_load)
        self.engine = QAMatchingEngine(self.kb)
        self.confidence_threshold = confidence_threshold
        self.conversation_history: List[Dict[str, str]] = []

    def _check_conversational_intent(self, normalized_query: str) -> Optional[str]:
        """Detects if query is a conversational statement or command."""
        # Exact match or word boundary check
        q = normalized_query.strip()

        if q in GREETING_WORDS or any(q.startswith(g + " ") for g in ["hi", "hello", "hey"]):
            return "greeting"

        if q in FAREWELL_WORDS:
            return "farewell"

        if any(phrase in q for phrase in ABOUT_PHRASES):
            return "about"

        if q in HELP_WORDS or q == "help":
            return "help"

        if q in TOPIC_WORDS:
            return "topics"

        if q in RANDOM_WORDS:
            return "random"

        if any(w in q.split() for w in THANKS_WORDS):
            return "thanks"

        return None

    def _format_entry_response(self, entry: KBEntry, confidence: float) -> str:
        """Constructs a clean, readable markdown response for a found answer."""
        difficulty_emoji = {
            "Beginner": "🟢",
            "Intermediate": "🟡",
            "Advanced": "🔴"
        }.get(entry.difficulty, "📘")

        lines = [
            f"### {entry.question}",
            f"**Category:** `{entry.category}` | **Level:** {difficulty_emoji} {entry.difficulty} | **Match Score:** `{int(confidence * 100)}%`",
            "",
            entry.answer,
            ""
        ]

        if entry.code_example:
            lines.extend([
                "#### 💻 Code Example:",
                "```python",
                entry.code_example.strip(),
                "```",
                ""
            ])

        return "\n".join(lines)

    def ask(self, query: str) -> BotResponse:
        """Processes user input in natural language and returns an appropriate response."""
        clean_input = query.strip()
        if not clean_input:
            return BotResponse(
                query=query,
                intent="unknown",
                found=False,
                text="Please enter a question or topic about Python programming! (Type `/help` to see options)"
            )

        norm_query = normalize_text(clean_input)
        intent = self._check_conversational_intent(norm_query)

        # 1. Handle Greetings
        if intent == "greeting":
            greetings = [
                "Hello! 👋 I'm your Personal Python QA Chatbot. What Python concept or question can I help you with today?",
                "Hi there! 🐍 Ready to explore Python programming? Ask me anything about syntax, data structures, OOP, or built-ins!",
                "Hey! Welcome. Ask me any question related to Python—from basic variables to the GIL and decorators!"
            ]
            reply = random.choice(greetings)
            self._record(clean_input, reply)
            return BotResponse(query=query, intent="greeting", found=True, confidence=1.0, text=reply)

        # 2. Handle Farewell
        if intent == "farewell":
            farewells = [
                "Goodbye! Happy coding in Python! 🚀 Feel free to return whenever you have questions.",
                "See you later! Keep building great Python applications! 🐍✨",
                "Bye! Remember: *Readability counts* (PEP 20). Have a wonderful day!"
            ]
            reply = random.choice(farewells)
            self._record(clean_input, reply)
            return BotResponse(query=query, intent="farewell", found=True, confidence=1.0, text=reply)

        # 3. Handle About / Identity
        if intent == "about":
            reply = (
                "🤖 **About Me:**\n\n"
                "I am your **Personal Python Question-Answering Chatbot**.\n"
                f"I am powered by a curated knowledge base of **{self.kb.count()} Python topics** spanning:\n"
                "- Core Syntax & PEP 8\n"
                "- Data Structures (Lists, Tuples, Sets, Dicts)\n"
                "- Functions, Lambdas, and Scope (LEGB)\n"
                "- Advanced Features (Decorators, Generators, Iterators)\n"
                "- Object-Oriented Programming (MRO, Dunder methods, Dataclasses)\n"
                "- Error & Exception Handling\n"
                "- Concurrency (GIL, Multiprocessing, Asyncio)\n\n"
                "Ask me any question in natural language (e.g. *'What is a decorator?'*, *'List vs tuple'*, *'Explain GIL'*)."
            )
            self._record(clean_input, reply)
            return BotResponse(query=query, intent="about", found=True, confidence=1.0, text=reply)

        # 4. Handle Help
        if intent == "help":
            reply = (
                "📖 **Python Chatbot Quick Guide:**\n\n"
                "- **Ask in plain English:** e.g., *'What is the difference between list and tuple?'*\n"
                "- **Search technical terms:** e.g., *'walrus operator'*, *'generators'*, *'MRO'*\n"
                "- **Available Commands:**\n"
                "  - `/topics` or `topics`: View all available categories\n"
                "  - `/random` or `random`: Get a random Python interview question & code snippet\n"
                "  - `/help`: Display this help guide\n"
                "  - `/exit` or `quit`: Exit the chatbot"
            )
            self._record(clean_input, reply)
            return BotResponse(query=query, intent="help", found=True, confidence=1.0, text=reply)

        # 5. Handle Topics
        if intent == "topics":
            categories = self.kb.get_categories()
            cat_lines = []
            for cat in categories:
                entries_in_cat = self.kb.get_by_category(cat)
                cat_lines.append(f"- **{cat}** ({len(entries_in_cat)} topics)")
            reply = (
                f"📚 **Available Python Topic Categories ({len(categories)} categories, {self.kb.count()} total topics):**\n\n"
                + "\n".join(cat_lines)
                + "\n\n*Tip: Ask a question about any of these topics!*"
            )
            self._record(clean_input, reply)
            return BotResponse(query=query, intent="topics", found=True, confidence=1.0, text=reply)

        # 6. Handle Random
        if intent == "random":
            entry = self.kb.get_random()
            if entry:
                reply = f"🎲 **Random Python Question:**\n\n" + self._format_entry_response(entry, 1.0)
                self._record(clean_input, reply)
                return BotResponse(
                    query=query,
                    intent="random",
                    found=True,
                    confidence=1.0,
                    text=reply,
                    entry=entry
                )

        # 7. Handle Thanks
        if intent == "thanks":
            reply = "You're very welcome! 😊 Always here to help you master Python. Have another question?"
            self._record(clean_input, reply)
            return BotResponse(query=query, intent="thanks", found=True, confidence=1.0, text=reply)

        # 8. Core QA Match via Hybrid NLP Engine
        matches: List[MatchResult] = self.engine.search(clean_input, top_k=3)

        if not matches or matches[0].score < self.confidence_threshold:
            # Fallback with smart suggestions
            suggestions = [m.entry.question for m in matches] if matches else [
                e.question for e in random.sample(self.kb.get_all(), min(3, self.kb.count()))
            ]

            lines = [
                "🤔 **I'm not completely certain I have an exact answer for that query.**",
                "",
                "Did you mean one of these Python topics?",
            ]
            for s in suggestions:
                lines.append(f"- {s}")
            lines.append("\n*Try rephrasing your question or type `/topics` to see what I know!*")

            reply = "\n".join(lines)
            self._record(clean_input, reply)
            return BotResponse(
                query=query,
                intent="qa",
                found=False,
                confidence=matches[0].score if matches else 0.0,
                text=reply,
                suggestions=suggestions
            )

        # Found confident match
        best_match = matches[0]
        related = [
            m.entry.question for m in matches[1:]
            if m.score >= 0.25 and m.entry.id != best_match.entry.id
        ]

        reply = self._format_entry_response(best_match.entry, best_match.score)

        if related:
            reply += "\n💡 **Related Questions:**\n"
            for rel in related:
                reply += f"- {rel}\n"

        self._record(clean_input, reply)
        return BotResponse(
            query=query,
            intent="qa",
            found=True,
            confidence=best_match.score,
            text=reply,
            entry=best_match.entry,
            related_questions=related
        )

    def _record(self, user_msg: str, bot_msg: str) -> None:
        """Maintains conversational context."""
        self.conversation_history.append({"user": user_msg, "bot": bot_msg})

    def get_history(self) -> List[Dict[str, str]]:
        """Return chat history."""
        return list(self.conversation_history)

    def clear_history(self) -> None:
        """Clear conversation history."""
        self.conversation_history.clear()


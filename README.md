# 🐍 Personal Python Question-Answering Chatbot

An intelligent, natural language question-answering chatbot dedicated to Python programming. The chatbot accepts natural language questions from developers, understands technical queries, and delivers comprehensive explanations along with runnable Python code examples based on a rich predefined knowledge base.

---

## 🌟 Key Features

- **Natural Language Question Answering:** Ask questions in plain conversational English (e.g., *"What is the difference between list and tuple?"*, *"Explain how generators work with yield"*, *"What does the walrus operator do?"*).
- **Comprehensive Predefined Knowledge Base:** 48+ thoroughly documented Python topics covering 8 categories:
  - 🟢 **Basics & Syntax:** PEP 8, Dynamic Typing, Docstrings, Type Hints, Control Flow (`pass`, `break`, `continue`), Walrus operator (`:=`).
  - 🟢 **Data Structures:** Lists, Tuples, Dictionaries, Sets, Comprehensions, Mutability, Shallow vs. Deep Copy, String formatting (`f-strings`), Sorting.
  - 🟢 **Operators & Logic:** Identity (`is`) vs. Equality (`==`), Truthy/Falsy values, Ternary conditionals.
  - 🟡 **Functions:** `*args` and `**kwargs`, Lambda functions, LEGB Variable Scope, Mutable default argument pitfall.
  - 🟡 **Advanced Functional Programming:** Decorators (`@functools.wraps`), Generators & `yield`, Iterators (`__iter__`, `__next__`), Generator expressions.
  - 🟡 **Object-Oriented Programming (OOP):** Classes, `__init__`, Instance vs. Class vs. Static methods, Inheritance & MRO, Dunder magic methods, `@property`, `@dataclass`, Duck Typing.
  - 🟡 **Error & Exception Handling:** `try...except...else...finally`, Custom user-defined exceptions.
  - 🟡 **File I/O & Context Managers:** `with` statement, Context managers, JSON serialization (`load`/`loads`, `dump`/`dumps`), `pathlib`.
  - 🔴 **Standard Library & Concurrency:** Global Interpreter Lock (GIL), Threading vs. Multiprocessing vs. Asyncio, Memory management & Garbage Collection, `collections`, `itertools`, `re` (regex).
- **Hybrid NLP Retrieval Engine:**
  - **Python-Aware Preprocessor:** Preserves code keywords (`is`, `not`, `in`, `def`, `class`, `self`) and special syntax (`:=`, `==`, dunders).
  - **TF-IDF + Cosine Similarity:** Vectorizes unigrams and bigrams with sublinear term-frequency scaling.
  - **Domain Tag & Keyword Boosting:** Scores boosted when domain-specific terms match question tags.
  - **Pattern & Direct Matching:** Exact and fuzzy token matching for common question phrasing.
  - **Confidence Thresholding & Fallback Suggestions:** If confidence is below threshold, the chatbot recommends the top 3 closest related questions rather than hallucinating.
- **Multiple User Interfaces:**
  1. 💻 **Interactive CLI:** Beautiful terminal interface with ANSI colors, special commands (`/topics`, `/random`, `/help`, `/clear`, `/exit`), and single-query execution.
  2. 🌐 **Streamlit Web UI:** Modern chat bubble interface, sidebar topic explorer, quick-question buttons, and syntax-highlighted code blocks.
  3. 🚀 **FastAPI REST API:** Full OpenAPI/Swagger-documented endpoints for integration into third-party apps.
- **Automated Test Suite:** 30 unit and integration tests passing with `pytest`.

---

## 🏗️ Architecture

```
User Query: "What is the difference between list and tuple?"
    │
    ▼
[Preprocessor & Normalizer]
    ├── Lowercasing & Punctuation Cleanup
    ├── Python Token Preservation (is, not, in, :=, ==, __dunders__)
    └── Stopword Filtering
    │
    ▼
[Hybrid Matching Engine]
    ├── Pattern & Exact Match Check
    ├── TF-IDF Vectorization ((1,2)-grams, sublinear TF)
    ├── Cosine Similarity Matrix Computation
    └── Domain Tag & Technical Keyword Boost
    │
    ▼
[Confidence Evaluator]
    ├── Score >= 0.30: Return Best Answer + Code Snippet + Related Topics
    └── Score < 0.30: Graceful Fallback with Top-3 Candidate Suggestions
    │
    ▼
[Presentation Layer]
    ├── CLI (Interactive Terminal / Single Query Mode)
    ├── Web UI (Streamlit Chat Application)
    └── REST API (FastAPI JSON Endpoints)
```

---

## 📁 Project Structure

```
c:\project5\
├── data/
│   └── python_kb.json             # Predefined Python Q&A knowledge base
├── src/
│   ├── __init__.py                # Package exports
│   ├── preprocessor.py            # Python-aware text normalization & tokenization
│   ├── knowledge_base.py          # Knowledge base loader, model & query methods
│   ├── engine.py                  # Hybrid TF-IDF + Cosine similarity search engine
│   ├── bot.py                     # Chatbot controller, intents & response formatting
│   └── build_kb.py                # Script to regenerate/extend knowledge base JSON
├── cli.py                         # Interactive & single-query Command Line Interface
├── app.py                         # Streamlit Web Chatbot UI
├── api.py                         # FastAPI RESTful API server
├── tests/
│   ├── __init__.py
│   ├── test_preprocessor.py       # Preprocessor tests
│   ├── test_knowledge_base.py     # Knowledge base validation tests
│   ├── test_engine.py             # NLP engine matching tests
│   ├── test_bot.py                # Bot controller & intent tests
│   └── test_api.py                # REST API endpoint tests
├── conftest.py                    # Pytest configuration
├── pytest.ini                     # Pytest settings
├── requirements.txt               # Dependencies
└── README.md                      # Project documentation
```

---

## 🚀 Getting Started

### 1. Prerequisites
- Python 3.9+ (Python 3.11 recommended)

### 2. Installation
Clone or navigate to the repository directory and install dependencies:
```bash
pip install -r requirements.txt
```

---

## 💻 Usage

### 1. Interactive Command-Line Interface (CLI)

Run the interactive terminal chatbot:
```bash
python cli.py
```

**Commands inside the interactive CLI:**
- Type any question directly: `What is a decorator?`
- `/topics` : View all available topic categories
- `/random` : Get a random Python interview question with code example
- `/help`   : View command list
- `/clear`  : Clear conversation history
- `/exit`   : Exit the chatbot

### 2. Single-Query CLI Mode

Query the chatbot directly from shell scripts or command line:
```bash
python cli.py -q "What is a decorator?"
python cli.py -q "Explain the Global Interpreter Lock (GIL)"
python cli.py --topics
python cli.py --random
```

### 3. Streamlit Web Interface

Launch the interactive web chatbot in your browser:
```bash
streamlit run app.py
```
This opens the web interface (default: `http://localhost:8501`) featuring:
- Conversational chat bubbles with role avatars
- Quick-question buttons on the sidebar
- Category explorer to browse questions by topic
- Difficulty badges (🟢 Beginner, 🟡 Intermediate, 🔴 Advanced)
- Code blocks with one-click copy functionality

### 4. FastAPI REST API

Start the REST API server:
```bash
python api.py
```
Or via uvicorn directly:
```bash
uvicorn api:app --host 127.0.0.1 --port 8000 --reload
```

Interactive Swagger documentation is available at: **`http://127.0.0.1:8000/docs`**

#### REST API Endpoints:
- **`GET /health`**: System status and knowledge base summary
- **`POST /ask`**: Ask a question
  ```bash
  curl -X POST "http://127.0.0.1:8000/ask" \
       -H "Content-Type: application/json" \
       -d "{\"query\": \"What is the difference between list and tuple?\"}"
  ```
- **`GET /topics`**: List all categories
- **`GET /questions`**: List all questions (optional: `?category=Data%20Structures`)
- **`GET /random`**: Fetch a random question and answer

---

## 🧪 Running Automated Tests

Run the complete test suite with `pytest`:
```bash
python -m pytest -v
```

All 30 unit and integration tests will execute and validate:
- ✅ Text preprocessing and Python keyword preservation
- ✅ Knowledge base schema, unique IDs, and field integrity
- ✅ Hybrid NLP engine similarity matching across phrasing variations
- ✅ Bot conversational intent parsing and fallback recommendation behavior
- ✅ REST API endpoints and error validation

---

## 💡 Example Queries to Try

| Question Category | Example Queries |
| :--- | :--- |
| **Basics** | *"What is Python?"*, *"What are PEP 8 guidelines?"*, *"Explain the walrus operator :="* |
| **Data Structures** | *"What is the difference between list and tuple?"*, *"How do list comprehensions work?"*, *"Shallow copy vs deep copy"* |
| **Logic & Operators** | *"Difference between is and =="*, *"What are truthy and falsy values?"* |
| **Functions** | *"What are *args and **kwargs?"*, *"Why is mutable default argument dangerous?"*, *"Explain LEGB scope"* |
| **Advanced** | *"What is a decorator and how to write one?"*, *"How do generators and yield work?"*, *"Iterators protocol"* |
| **OOP** | *"What is __init__ and self?"*, *"Explain Method Resolution Order (MRO)"*, *"What are dataclasses?"* |
| **Concurrency** | *"What is the Global Interpreter Lock (GIL)?"*, *"Threading vs Multiprocessing vs Asyncio"* |
| **Chatbot Commands** | *"Hello"*, *"Who are you?"*, *"Help"*, *"Topics"*, *"Random"*, *"Bye"* |

---

## 🛠️ Extending the Knowledge Base

To add new topics:
1. Open `src/build_kb.py`.
2. Add a new dictionary entry to the `KNOWLEDGE_BASE` list:
   ```python
   {
       "id": "my_new_topic",
       "category": "Data Structures",
       "difficulty": "Intermediate",
       "question": "What is ...?",
       "patterns": ["How does ... work?", "... in Python"],
       "tags": ["tag1", "tag2"],
       "summary": "One line summary...",
       "answer": "Detailed markdown explanation...",
       "code_example": "# Code snippet\n..."
   }
   ```
3. Run `python src/build_kb.py` to regenerate `data/python_kb.json`.
4. Run `python -m pytest` to verify test passing.

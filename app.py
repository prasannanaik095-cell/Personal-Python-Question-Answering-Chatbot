"""Streamlit Web Application for the Personal Python QA Chatbot.

Provides an interactive chat interface, topic explorer, sample question chips,
and real-time natural language question answering with confidence metrics.
"""

import streamlit as st
from src.bot import PythonQABot, BotResponse
from src.knowledge_base import KBEntry

# Page Configuration
st.set_page_config(
    page_title="Python QA Chatbot",
    page_icon="🐍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E88E5;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #555555;
        margin-bottom: 1.5rem;
    }
    .badge-beginner {
        background-color: #E8F5E9;
        color: #2E7D32;
        padding: 3px 8px;
        border-radius: 12px;
        font-size: 0.8rem;
        font-weight: 600;
    }
    .badge-intermediate {
        background-color: #FFF8E1;
        color: #F57F17;
        padding: 3px 8px;
        border-radius: 12px;
        font-size: 0.8rem;
        font-weight: 600;
    }
    .badge-advanced {
        background-color: #FFEBEE;
        color: #C62828;
        padding: 3px 8px;
        border-radius: 12px;
        font-size: 0.8rem;
        font-weight: 600;
    }
    .category-tag {
        background-color: #E3F2FD;
        color: #1565C0;
        padding: 3px 8px;
        border-radius: 12px;
        font-size: 0.8rem;
        font-weight: 600;
    }
    .stCodeBlock {
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)


# Initialize Bot in Session State
@st.cache_resource
def get_bot():
    """Cache bot instance to avoid reloading TF-IDF index on every turn."""
    return PythonQABot()


bot = get_bot()

# Initialize Chat History in Session State
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "👋 **Hello!** I am your Personal Python Question-Answering Chatbot.\n\nAsk me any question in natural language about Python programming (syntax, lists, dicts, OOP, decorators, GIL, async, and more)!",
            "meta": None
        }
    ]

# Sidebar Controls
with st.sidebar:
    st.markdown("### 🐍 Python QA Assistant")
    st.markdown(f"**Knowledge Base:** `{bot.kb.count()} Topics` | `{len(bot.kb.get_categories())} Categories`")
    st.divider()

    st.markdown("#### ⚡ Quick Questions")
    sample_queries = [
        "What is the difference between list and tuple?",
        "What is a decorator and how to use it?",
        "Explain Global Interpreter Lock (GIL)",
        "What is the walrus operator := ?",
        "Why is mutable default argument bad?",
        "Difference between is and ==",
        "How do generators and yield work?",
        "Asyncio vs Threading vs Multiprocessing"
    ]

    selected_sample = None
    for q in sample_queries:
        if st.button(f"💬 {q}", key=f"btn_{q}", use_container_width=True):
            selected_sample = q

    st.divider()

    # Topic Category Explorer
    st.markdown("#### 📚 Topic Explorer")
    categories = ["All Categories"] + bot.kb.get_categories()
    selected_cat = st.selectbox("Filter by Category", categories)

    if selected_cat != "All Categories":
        cat_entries = bot.kb.get_by_category(selected_cat)
        st.markdown(f"**{len(cat_entries)} Questions in this category:**")
        for e in cat_entries:
            if st.button(f"• {e.question}", key=f"cat_q_{e.id}", use_container_width=True):
                selected_sample = e.question

    st.divider()

    if st.button("🗑️ Clear Conversation", use_container_width=True):
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": "👋 Conversation cleared! What Python question would you like to explore next?",
                "meta": None
            }
        ]
        bot.clear_history()
        st.rerun()

    st.markdown("---")
    with st.expander("ℹ️ How It Works"):
        st.markdown("""
        **Architecture:**
        - **NLP Engine:** Scikit-Learn TF-IDF vectorization with $(1, 2)$-gram feature extraction.
        - **Similarity:** Cosine distance computation on normalized textual corpus.
        - **Keyword Boosting:** Technical tag boosting for programming constructs.
        - **Fallback:** Top-3 nearest candidate recommendations if confidence falls below threshold.
        """)

# Header
st.markdown('<div class="main-header">🐍 Personal Python Question-Answering Chatbot</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Ask questions in plain English and get comprehensive answers with runnable code examples.</div>', unsafe_allow_html=True)

# Render Chat History
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        meta = msg.get("meta")
        if meta and meta.get("entry"):
            entry_dict = meta["entry"]
            with st.expander("💻 Run / Copy Code Example", expanded=True):
                st.code(entry_dict["code_example"], language="python")


def handle_user_query(user_query: str):
    """Processes query, gets bot response, and renders it in the conversation."""
    # Append user message
    st.session_state.messages.append({"role": "user", "content": user_query, "meta": None})

    # Get Bot Response
    response: BotResponse = bot.ask(user_query)

    # Format assistant reply
    meta_info = None
    if response.found and response.entry:
        diff_class = {
            "Beginner": "badge-beginner",
            "Intermediate": "badge-intermediate",
            "Advanced": "badge-advanced"
        }.get(response.entry.difficulty, "badge-beginner")

        score_pct = int(response.confidence * 100)

        header_html = (
            f"<div style='margin-bottom: 8px;'>"
            f"<span class='category-tag'>📂 {response.entry.category}</span> "
            f"<span class='{diff_class}'>● {response.entry.difficulty}</span> "
            f"<span style='font-size: 0.8rem; color: #666; margin-left: 8px;'>Match Confidence: <b>{score_pct}%</b></span>"
            f"</div>"
        )

        content = f"{header_html}\n\n### {response.entry.question}\n\n{response.entry.answer}"

        if response.related_questions:
            content += "\n\n💡 **Related Questions you might ask:**\n"
            for rel in response.related_questions:
                content += f"- {rel}\n"

        meta_info = {"entry": response.entry.to_dict(), "confidence": response.confidence}

    elif response.intent in ("greeting", "farewell", "about", "help", "topics", "thanks"):
        content = response.text
    else:
        # Fallback
        content = response.text
        meta_info = {"suggestions": response.suggestions}

    st.session_state.messages.append({
        "role": "assistant",
        "content": content,
        "meta": meta_info
    })


# Process input from sidebar button click if any
if selected_sample:
    handle_user_query(selected_sample)
    st.rerun()

# User Input from chat bar
if prompt := st.chat_input("Ask a Python question (e.g. 'What is a list comprehension?', 'Explain *args and **kwargs')..."):
    handle_user_query(prompt)
    st.rerun()


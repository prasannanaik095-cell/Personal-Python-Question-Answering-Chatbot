"""Command-Line Interface for the Personal Python QA Chatbot.

Supports interactive conversation mode with ANSI formatting and commands,
as well as direct single-query execution from the command line.
"""

import argparse
import sys
from typing import Optional

from src.bot import PythonQABot, BotResponse

# ANSI Terminal Color Helpers
class Colors:
    HEADER = "\033[95m"
    BLUE = "\033[94m"
    CYAN = "\033[96m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"
    DIM = "\033[2m"
    RESET = "\033[0m"


def print_banner(kb_count: int, cat_count: int) -> None:
    """Displays welcome banner."""
    banner = f"""{Colors.CYAN}{Colors.BOLD}
  ╔═════════════════════════════════════════════════════════════════════════╗
  ║                🐍 PERSONAL PYTHON QA CHATBOT 🐍                        ║
  ║         Ask anything about Python syntax, OOP, GIL, and more!          ║
  ║         Loaded {kb_count} topics across {cat_count} categories                           ║
  ╚═════════════════════════════════════════════════════════════════════════╝{Colors.RESET}
{Colors.DIM}Type your question in natural language. Special commands:
  /help    - Show commands help
  /topics  - Browse all topic categories
  /random  - Get a random Python interview question
  /clear   - Clear conversation history
  /exit    - Exit the chatbot
{Colors.RESET}"""
    print(banner)


def display_bot_response(response: BotResponse) -> None:
    """Formats and prints the bot response cleanly in the terminal."""
    print()
    if response.found and response.entry:
        diff_color = {
            "Beginner": Colors.GREEN,
            "Intermediate": Colors.YELLOW,
            "Advanced": Colors.RED
        }.get(response.entry.difficulty, Colors.BLUE)

        print(f"{Colors.GREEN}{Colors.BOLD}🤖 Bot:{Colors.RESET}")
        print(f"{Colors.BOLD}{Colors.CYAN}Q: {response.entry.question}{Colors.RESET}")
        print(f"{Colors.DIM}Category: {response.entry.category} | Level: {diff_color}{response.entry.difficulty}{Colors.RESET} | Match: {int(response.confidence * 100)}%{Colors.RESET}")
        print("─" * 70)
        print(response.entry.answer)

        if response.entry.code_example:
            print(f"\n{Colors.YELLOW}{Colors.BOLD}💻 Code Example:{Colors.RESET}")
            print(f"{Colors.DIM}```python{Colors.RESET}")
            for line in response.entry.code_example.strip().split("\n"):
                print(f"  {line}")
            print(f"{Colors.DIM}```{Colors.RESET}")

        if response.related_questions:
            print(f"\n{Colors.CYAN}{Colors.BOLD}💡 Related Questions:{Colors.RESET}")
            for rel in response.related_questions:
                print(f"  • {rel}")

    elif response.intent in ("greeting", "farewell", "about", "help", "topics", "thanks"):
        print(f"{Colors.GREEN}{Colors.BOLD}🤖 Bot:{Colors.RESET}")
        print(response.text)

    else:
        # Fallback / not found
        print(f"{Colors.YELLOW}{Colors.BOLD}🤖 Bot:{Colors.RESET}")
        print(f"{Colors.YELLOW}I couldn't find a direct answer with high confidence.{Colors.RESET}")
        if response.suggestions:
            print(f"\n{Colors.CYAN}Did you mean one of these topics?{Colors.RESET}")
            for s in response.suggestions:
                print(f"  • {s}")
        print(f"\n{Colors.DIM}Tip: Try rephrasing or type /topics to explore.{Colors.RESET}")

    print("\n" + "═" * 70)


def run_interactive(bot: PythonQABot) -> None:
    """Runs interactive command-line session."""
    print_banner(bot.kb.count(), len(bot.kb.get_categories()))

    while True:
        try:
            prompt = f"\n{Colors.BOLD}{Colors.BLUE}You > {Colors.RESET}"
            user_input = input(prompt).strip()

            if not user_input:
                continue

            # Command checks
            cmd = user_input.lower()
            if cmd in ("/exit", "/quit", "exit", "quit"):
                farewell = bot.ask("bye")
                print(f"\n{Colors.GREEN}{farewell.text}{Colors.RESET}\n")
                break

            if cmd in ("/clear", "clear"):
                bot.clear_history()
                print(f"{Colors.DIM}Conversation history cleared.{Colors.RESET}")
                continue

            response = bot.ask(user_input)
            display_bot_response(response)

        except (KeyboardInterrupt, EOFError):
            print(f"\n\n{Colors.YELLOW}Session interrupted. Goodbye! 👋{Colors.RESET}")
            sys.exit(0)


def main():
    parser = argparse.ArgumentParser(
        description="Personal Python Question-Answering Chatbot",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  python cli.py                              # Start interactive chatbot
  python cli.py -q "What is a decorator?"    # Ask a single question directly
  python cli.py --topics                     # List all available topics
  python cli.py --random                     # Display a random Python QA
"""
    )
    parser.add_argument("-q", "--query", type=str, help="Single question to ask the chatbot directly")
    parser.add_argument("--topics", action="store_true", help="List all available topic categories")
    parser.add_argument("--random", action="store_true", help="Get a random Python Q&A")
    parser.add_argument("--kb", type=str, help="Custom path to knowledge base JSON file")

    args = parser.parse_args()
    bot = PythonQABot(kb_path=args.kb)

    if args.topics:
        res = bot.ask("/topics")
        print(res.text)
        return

    if args.random:
        res = bot.ask("/random")
        display_bot_response(res)
        return

    if args.query:
        res = bot.ask(args.query)
        display_bot_response(res)
        return

    # Default to interactive mode
    run_interactive(bot)


if __name__ == "__main__":
    main()


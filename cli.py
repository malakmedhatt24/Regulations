"""Terminal chat.  Run with:  python cli.py   (or: python cli.py "your question")"""

import sys

from regbot.pipeline import RegBot


def ask(bot: RegBot, question: str, history: list[dict]) -> str:
    hits, chunks = bot.answer(question, history)
    reply = ""
    for piece in chunks:
        reply += piece
        print(piece, end="", flush=True)
    print("\n\nSources:")
    for n, h in enumerate(hits, start=1):
        print(f"  [{n}] {h.chunk.citation}")
    return reply


def main() -> None:
    bot = RegBot()
    if len(sys.argv) > 1:
        ask(bot, " ".join(sys.argv[1:]), [])
        return

    print("Regulations assistant. Ask a question, or type 'exit' to quit.\n")
    history: list[dict] = []
    while True:
        try:
            question = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if question.lower() in {"exit", "quit"}:
            break
        if not question:
            continue
        print("\nBot: ", end="")
        reply = ask(bot, question, history)
        history += [{"role": "user", "content": question}, {"role": "assistant", "content": reply}]
        print()


if __name__ == "__main__":
    main()

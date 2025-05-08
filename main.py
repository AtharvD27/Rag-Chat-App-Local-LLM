# main.py
import argparse
from utils import load_config
from run_chat import (
    load_documents, update_vectorstore,
    setup_llm, handle_session, start_session
)

def main():
    parser = argparse.ArgumentParser(description="RAG Chat Interface")
    parser.add_argument("--model_path", type=str, help="Override model path")
    parser.add_argument("--temperature", type=float, help="Override LLM temperature")
    parser.add_argument("--config", type=str, default="config.yaml", help="Config file path")
    parser.add_argument("--skip_update", action="store_true", help="Skip vectorstore update")

    args = parser.parse_args()

    config = load_config(args.config)
    overrides = {
        "model_path": args.model_path,
        "temperature": args.temperature
    }

    # Chat session
    print("🤖 Starting RAG chat agent...")
    chunks = load_documents(config)
    retriever = update_vectorstore(config, chunks, skip_update=args.skip_update)
    llm = setup_llm(config, overrides)

    config["retriever"] = retriever
    config["llm_instance"] = llm

    snap, memory, session_id = handle_session(config)
    agent = start_session(config, memory)

    print("\n🔍 Ask questions. Type 'exit' to quit.\n")
    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in ["exit", "quit"]:
            break
        answer, sources = agent.ask(user_input)
        print(f"\n💬 Answer:\n{answer}\n📚 Sources:")
        for s in sources:
            print(f" - {s['file']} (Page {s['page']}, Chunk {s['chunk']}): {s['text'][:150]}...\n")
        snap.record_turn(user_input, answer, sources)
        snap.save_snapshot()

    print("✅ Exiting. Final snapshot saved.\n")

if __name__ == "__main__":
    main()

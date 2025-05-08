from document_loader import PDFDirectoryLoader
from vectorstore_manager import VectorstoreManager
from chat_agent import ChatAgent
from snapshot_manager import SnapshotManager
from get_llm import get_local_llm
from langchain.memory import ConversationBufferMemory

def print_sessions(sessions):
    print("\n🗂️ Available Sessions:")
    for i, session in enumerate(sessions):
        print(f"{i+1}. Alias: {session['alias']}")
        print(f"   ID: {session['id']}")
        print(f"   Time: {session['timestamp']}")
        print(f"   Preview: {session['first_msg'][:80]}...\n")

# Step 1: Load documents and chunks
loader = PDFDirectoryLoader(path="./data")
documents = loader.load()
chunks = loader.split_documents(documents)


# Step 2: Load vectorstore and update only if needed
vs_manager = VectorstoreManager(config_path="config.yaml")
vs_manager.load_vectorstore()

if vs_manager.needs_update(chunks):
    print("🔁 Updating vectorstore with new documents...")
    vs_manager.add_documents(chunks)
else:
    print("✅ Vectorstore is up to date.")

retriever = vs_manager.vs.as_retriever(search_kwargs={"k": 3})
llm = get_local_llm()


# Step 3: Handle session
snap = SnapshotManager()

print("\n🎯 Choose an option:")
print("1: Start new session")
print("2: Resume existing session")
print("3: Resume latest session")
choice = input("Choice: ").strip()

# Start new session
if choice == "1":
    alias = input("Enter optional session alias (or press Enter to skip): ").strip()
    alias = alias if alias else None
    session_id = snap.start_new_session()
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    print(f"🆕 Started session {session_id} (alias: {alias or session_id})")

# Resume existing session    
elif choice == "2":
    sessions = snap.list_sessions()
    if not sessions:
        print("❌ No sessions found. Starting new session.")
        session_id = snap.start_new_session()
        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    else:
        print_sessions(sessions)
        session_key = input("Enter alias or session ID to resume: ").strip()
        memory = snap.resume_session(session_key)
        if memory is None:
            print("⚠️ Invalid ID/alias. Starting new session.")
            session_id = snap.start_new_session()
            memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        else:
            session_id = snap.session_id
            print(f"♻️ Resumed session {session_id}")

#Resume latest session            
elif choice == "3":
    memory = snap.resume_latest()
    if memory is None:
        print("⚠️ No previous sessions found. Starting new one.")
        session_id = snap.start_new_session()
        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    else:
        session_id = snap.session_id
        print(f"⏪ Resumed latest session {session_id}")
        
else:
    print("❌ Invalid input, exiting.")
    exit(1)

# Step 4: Start agent
agent = ChatAgent(llm=llm, retriever=retriever, memory=memory)

print("\n🔍 Ask questions. Type 'exit' to quit.\n")
while True:
    user_input = input("You: ").strip()
    if user_input.lower() in ["exit", "quit"]:
        break
    answer, sources = agent.ask(user_input)

    print(f"\n💬 Answer:\n{answer}\n")
    print("📚 Sources:")
    for s in sources:
        print(f" - {s['file']} (Page {s['page']}, Chunk {s['chunk']}): {s['text'][:200]}...\n")

    snap.record_turn(user_input, answer, sources)
    snap.save_snapshot()

print("✅ Exiting. Final snapshot saved.\n")

import argparse
from tqdm import tqdm
from document_loader import PDFDirectoryLoader
from vectorstore_manager import VectorstoreManager

# CLI setup
parser = argparse.ArgumentParser(description="Manage vectorstore lifecycle.")
parser.add_argument("--update", action="store_true", help="Update vectorstore with only new documents.")
parser.add_argument("--delete", action="store_true", help="Delete the existing vectorstore.")
parser.add_argument("--reset", action="store_true", help="Delete and rebuild the vectorstore.")
parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file.")
args = parser.parse_args()

# Block silent execution with no flags
if not (args.update or args.delete or args.reset):
    parser.print_help()
    exit(0)

# Load manager
vs_manager = VectorstoreManager(config_path=args.config)

# DELETE ONLY
if args.delete:
    vs_manager.delete_vectorstore()
    exit(0)

# RESET (delete + full rebuild)
if args.reset:
    print("🔄 Resetting vectorstore...")
    vs_manager.delete_vectorstore()
    # Proceed to re-add documents after reset

# UPDATE or RESET both continue here
data_path = vs_manager.config.get("data_path", "./data")
loader = PDFDirectoryLoader(path=data_path, config_path=args.config)

# Load PDFs
documents = loader.load()
print(f"📄 Loaded {len(documents)} documents.")

# Split into chunks with progress
print("🔪 Splitting into chunks...")
chunks = []
for doc in tqdm(documents, desc="Chunking"):
    chunks.extend(loader.split_documents([doc]))

# Add to vectorstore
vs_manager.load_vectorstore()
vs_manager.add_documents(chunks)

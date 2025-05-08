import argparse
from document_loader import PDFDirectoryLoader
from vectorstore_manager import VectorstoreManager

# CLI setup
parser = argparse.ArgumentParser(description="Manage vectorstore lifecycle.")
parser.add_argument("--update", action="store_true", help="Update vectorstore with only new documents.")
parser.add_argument("--delete", action="store_true", help="Delete the existing vectorstore.")
parser.add_argument("--reset", action="store_true", help="Delete and rebuild the vectorstore.")
parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file.")
args = parser.parse_args()

# Load config
vs_manager = VectorstoreManager(config_path=args.config)

# DELETE
if args.delete:
    vs_manager.delete_vectorstore()
    exit(0)

# RESET
if args.reset:
    vs_manager.delete_vectorstore()
    # continue to full rebuild

# LOAD DOCUMENTS
loader = PDFDirectoryLoader(path=vs_manager.config.get("data_path", "./data"))
documents = loader.load()
chunks = loader.split_documents(documents)

# UPDATE
vs_manager.load_vectorstore()
vs_manager.add_documents(chunks)

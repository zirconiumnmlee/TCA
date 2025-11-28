import os
import hashlib
import json, csv
from tqdm import tqdm
import argparse
from pathlib import Path
from typing import List, Dict, Any
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "../../.."))
from config import Config
from src.llm.embedding import get_embedding

parser = argparse.ArgumentParser()
parser.add_argument(
    "--dataset",
    type=str,
    default="test",
    choices=["hotpotqa", "hotpotqa_test", "2wiki", "2wiki_test", "musique", "musique_test"],
    help="Dataset to use",
)
args = parser.parse_args()
config = Config(dataset=args.dataset)

def chunk_documents(docs: List[Dict[str, Any]], chunk_size: int = 1000, chunk_overlap: int = 200) -> List[Document]:
    """
    Split documents into chunks

    Args:
        contents: List of content(dict(content,metadata))
        chunk_size: Chunk size
        chunk_overlap: Chunk overlap size

    Returns:
        List[Document]: Split document chunks
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""]
    )

    documents = []

    for doc in docs:
        # Create Document object
        doc = Document(
            page_content=doc['content'],
            metadata=doc['metadata']
        )

        # Split document
        chunks = text_splitter.split_documents([doc])

        # Add unique identifier for each chunk
        for i, chunk in enumerate(chunks):
            chunk_hash = hashlib.md5(chunk.page_content.encode('utf-8')).hexdigest()
            chunk.metadata.update({
                'chunk_index': i,
                'total_chunks': len(chunks),
                'hash': chunk_hash
            })

        documents.extend(chunks)

    print(f"Document splitting completed, generated {len(documents)} chunks")
    return documents


def build_vector_store(documents: List[Document], output_dir: str):
    """
    Build vector database index

    Args:
        documents: List of document chunks
        config: Configuration object
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Get embedding client
    embedding_client = get_embedding(config)

    # Create Chroma vector store
    vector_store = Chroma(
        collection_name="Chunk",
        embedding_function=embedding_client,
        persist_directory=config.output_vectorDB_storage_path,
    )

    # Add documents in batches to avoid adding too many at once
    # Set batch size to respect API limits (max 64 for SiliconFlow)
    batch_size = 64
    total_docs = len(documents)

    for i in range(0, total_docs, batch_size):
        batch = documents[i:i+batch_size]
        vector_store.add_documents(batch)
        print(f"Processed {min(i+batch_size, total_docs)}/{total_docs} documents")

    print(f"Vector database construction completed! Index saved at: {output_dir}")
    return vector_store


def main():
    input_dir = config.input_corpus_dir
    output_dir = config.output_dir

    if os.path.exists(output_dir):
        import shutil
        shutil.rmtree(output_dir)
        print(f"🧹 Cleared existing vector database at {output_dir}")


    if args.dataset == "hotpotqa":
        with open(os.path.join(input_dir, "hotpot_dev_200.json")) as f:
            dev_dataset = json.load(f)

        dev = []
        for data in dev_dataset:
            dev.extend(data['context'])

        docs = []

        for d in dev:
            docs.append(
                {
                    "content": "".join(d[1]),
                    "metadata": {
                        "title":d[0]
                    }
                }
            )

    elif args.dataset == "hotpotqa_test":
        with open(os.path.join(input_dir, "hotpot_test_200.json")) as f:
            test_dataset = json.load(f)

        test = []
        for data in test_dataset:
            test.extend(data['context'])

        docs = []

        for t in test:
            docs.append(
                {
                    "content": "".join(t[1]),
                    "metadata": {
                        "title":t[0]
                    }
                }
            )

    elif args.dataset == "2wiki":
        with open(os.path.join(input_dir, "2wiki_dev_200.json")) as f:
            dev_dataset = json.load(f)

        dev = []
        for data in dev_dataset:
            dev.extend(data['context'])

        docs = []

        for d in dev:
            docs.append(
                {
                    "content": "".join(d[1]),
                    "metadata": {
                        "title":d[0]
                    }
                }
            )

    elif args.dataset == "2wiki_test":
        with open(os.path.join(input_dir, "2wiki_test_200.json")) as f:
            test_dataset = json.load(f)

        test = []
        for data in test_dataset:
            test.extend(data['context'])

        docs = []

        for t in test:
            docs.append(
                {
                    "content": "".join(t[1]),
                    "metadata": {
                        "title":t[0]
                    }
                }
            )

    elif args.dataset == "musique":
        with open(os.path.join(input_dir, "musique_dev_200.json")) as f:
            dev_dataset = json.load(f)

        dev = []
        for data in dev_dataset:
            dev.extend(data['paragraphs'])

        docs = []

        for d in dev:
            docs.append(
                {
                    "content": d["paragraph_text"],
                    "metadata": {
                        "title":d["title"]
                    }
                }
            )

    elif args.dataset == "musique_test":
        with open(os.path.join(input_dir, "musique_test_200.json")) as f:
            test_dataset = json.load(f)

        test = []
        for data in test_dataset:
            test.extend(data['paragraphs'])

        docs = []

        for t in test:
            docs.append(
                {
                    "content": t['paragraph_text'],
                    "metadata": {
                        "title":t["title"]
                    }
                }
            )

    print("=== Starting Vector Index Construction ===")
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")

    # 1. Split documents
    print("\nSplitting documents...")
    documents = chunk_documents(docs)

    # 2. Build vector store
    print("\nBuilding vector store...")
    vector_store = build_vector_store(documents, output_dir)

    # 3. Verify index
    print("\nVerifying index...")
    try:
        test_query = "test query"
        results = vector_store.similarity_search(test_query, k=3)
        print(f"Verification successful! Index contains {vector_store._collection.count()} documents")
        print(f"Test query returned {len(results)} results")
    except Exception as e:
        print(f"Verification failed: {e}")

    print("\n=== Vector Index Construction Completed ===")


if __name__ == "__main__":
    main()
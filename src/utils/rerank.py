from langchain_chroma import Chroma
from langchain_core.documents import Document
from config import Config
from src.llm.embedding import get_embedding
import time
import requests

session = requests.Session()
session.headers.update({"User-Agent": "langchain-rerank-client/1.0"})

SMART_TOKENIZER = None


def rerank(
    query,
    documents,
    model="BAAI/bge-reranker-v2-m3",
    top_n=5,
    return_documents=False,
    max_chunks_per_doc=4050,
    overlap_tokens=80,
    with_score=False,
    sleep_sec=0.15,
):
    """
    Rerank a list of documents with respect to the query.
    Returns indices (and optional scores) of the top-N most relevant docs.
    """
    url = "https://www.dmxapi.cn/v1/rerank"

    if isinstance(documents[0], str):
        docs_payload = [{"text": d} for d in documents]
    elif isinstance(documents[0], dict) and "text" in documents[0]:
        docs_payload = documents
    else:  # LangChain Document objects
        docs_payload = [{"text": d.page_content} for d in documents]

    payload = {
        "model": model,
        "query": query,
        "documents": docs_payload,
        "top_n": top_n,
        "return_documents": return_documents,
        "max_chunks_per_doc": max_chunks_per_doc,
        "overlap_tokens": overlap_tokens,
    }

    headers = {
        "Authorization": "Bearer YOUR_KEY",
        "Content-Type": "application/json",
    }

    # Rate-limiting
    if sleep_sec > 0:
        time.sleep(sleep_sec)

    try:
        resp = session.post(url, json=payload, headers=headers, timeout=30)
        if resp.status_code != 200:
            return _default(top_n, documents, with_score)

        data = resp.json()
        if "results" not in data:
            return _default(top_n, documents, with_score)

        rank = [x["index"] for x in data["results"]]
        score = [x["relevance_score"] for x in data["results"]]
        return (rank, score) if with_score else rank

    except Exception as exc:
        print(f"[rerank] API call failed: {exc}")
        return _default(top_n, documents, with_score)


def _default(top_n, documents, with_score):
    """Fallback: return original order with dummy scores."""
    default_rank = list(range(min(top_n, len(documents))))
    default_score = [1.0] * len(default_rank)
    return (default_rank, default_score) if with_score else default_rank

from typing import Dict, Any, List
from .base import Tool, ToolType, ToolsRegistry
from src.utils.retrieving_utils import (
    get_topk_similar_chunks,
    get_chunks_by_exact_match,
    get_chunks_by_fuzzy_match
)
from src.llm.llm import CachedChatOpenAI

# Dense search
@ToolsRegistry.register
class GetTopKSimilarChunksTool(Tool):
    name = "get_topk_similar_chunks"
    description = (
        "Retrieve top-K similar text chunks through vector database queries and semantic "
        "similarity calculations, combined with reranking. Suitable for scenarios requiring "
        "semantic relevance filtering from databases. Function returns a string containing "
        "a list of top-K similar text chunks."
    )
    tool_type = ToolType.RETRIEVING
    execute_input_description = {
        "query": "Natural language question or description of what you're looking for.",
    }
    execute_output_description = {"answer": "List of semantically relevant text chunks, ordered from most to least relevant"}

    def __init__(self, agent):
        self.agent = agent

    async def execute(self, query: str, topk: int=5, node_llm: CachedChatOpenAI=None) -> Dict[str, Any]:
        try:
            result = await get_topk_similar_chunks(
                query, topk, self.agent.config, node_llm, self.agent.used_hashes
            )
            return {"answer": result}
        except Exception as e:
            return {
                "answer": f"Error retrieving top-K similar chunks: {str(e)}",
                "error": True,
            }


@ToolsRegistry.register
class GetChunksByExactMatchTool(Tool):
    name = "get_chunks_by_exact_match"
    description = (
        "Find text chunks containing the EXACT keyword or phrase (case-insensitive). "
        "The keyword must appear verbatim in the text. Best for: finding specific terms, "
        "technical keywords, proper nouns, code identifiers, or when you need to verify "
        "if specific text exists. "
    )
    tool_type = ToolType.RETRIEVING
    execute_input_description = {
        "keyword": "**Word or short phrase** to exat match",
    }
    execute_output_description = {"answer": "List of text chunks with exact matches"}

    def __init__(self, agent):
        self.agent = agent

    async def execute(self, keyword: str, node_llm: CachedChatOpenAI) -> Dict[str, Any]:
        try:
            result = await get_chunks_by_exact_match(
                keyword, self.agent.config, node_llm, self.agent.used_hashes
            )
            # Convert result(list) to string
            return {"answer": result}
        except Exception as e:
            return {"answer": f"Exact match query error: {str(e)}", "error": True}


@ToolsRegistry.register
class GetChunksByFuzzyMatchTool(Tool):
    name = "get_chunks_by_fuzzy_match"
    description = (
        "Find text chunks using flexible pattern matching that tolerates typos, variations, "
        "and partial matches. Combines lexical similarity (character/word overlap) with semantic "
        "understanding. Best for: handling misspellings, finding content when unsure of exact wording, "
        "matching similar terms, or when exact match returns nothing. "
        "More permissive than exact match but more keyword-focused than semantic search. "
        "Returns matches ranked by similarity score."
    )
    tool_type = ToolType.RETRIEVING
    execute_input_description = {
        "keyword": "**Word or short phrase** to fuzzy match",
    }
    execute_output_description = {"answer": "List of text chunks with fuzzy matches"}

    def __init__(self, agent):
        self.agent = agent

    async def execute(self, keyword: str, node_llm: CachedChatOpenAI) -> Dict[str, Any]:
        try:
            result = await get_chunks_by_fuzzy_match(
                keyword, self.agent.config, node_llm, self.agent.used_hashes
            )
            return {"answer": result}
        except Exception as e:
            return {"answer": f"Fuzzy match query error: {str(e)}", "error": True}

# Sparse retrieve
from elasticsearch import Elasticsearch

@ToolsRegistry.register
class GetChunksByBM25Tool(Tool):
    name = "get_chunks_by_bm25"
    description = (
        "Retrieve documents using BM25 algorithm, a term-frequency based ranking method. "
        "Prioritizes documents where query terms appear frequently but penalizes common words. "
        "Best for: keyword-rich queries, when you know important terms, factoid questions, "
        "or when you want traditional 'search engine' behavior. Works well with longer queries "
        "containing multiple specific terms."
    )
    tool_type = ToolType.RETRIEVING
    execute_input_description = {
        "query": "Search query for sparse retrieval.",
    }
    execute_output_description = {"answer": "List of top-k BM25 documents."}

    def __init__(self, agent):
        self.agent = agent
        self.es = Elasticsearch(hosts="http://localhost:9200")

    async def execute(self, query: str, topk: int = 5, **kwargs) -> Dict[str, Any]:
        try:
            res = self.es.search(
                index=self.agent.config.dataset,
                body={
                    "query": {"multi_match": {"query": query, "fields": ["title", "text"]}},
                    "size": topk
                }
            )
            docs = [hit["_source"]["text"] for hit in res["hits"]["hits"]]
            return {"answer": docs}
        except Exception as e:
            return {"answer": f"BM25 query error: {str(e)}", "error": True}

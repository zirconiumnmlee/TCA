import re
from langchain_chroma import Chroma
from langchain_core.documents import Document
from config import Config
from ..llm import CachedChatOpenAI
from src.llm.embedding import get_embedding
import requests
import json
from typing import List, Set
from difflib import SequenceMatcher
from rerank import rerank


# ========== Memory retrieval utils ==========
def get_topk_tool_experience(
    query: str,
    topk: int=5,
    config: Config=None,
) -> List[str]:

    if not query or not query.strip():
        raise ValueError("Query cannot be empty or whitespace")

    if topk <= 0:
        raise ValueError("topk must be a positive integer")

    if config is None:
        raise ValueError("config cannot be None")

    try:
        embedding_client = get_embedding(config)
        vector_store = Chroma(
            collection_name="TrajectoryMemory",
            embedding_function=embedding_client,
            persist_directory=config.trajectory_memory_vectorDB_storage_path,
        )
    except Exception as e:
        return []

    try:
        search_k = min(max(topk * 4, 20), 100)
        retriever = vector_store.as_retriever(search_kwargs={"k": search_k})
        results = retriever.invoke(query)
    except Exception as e:
        return []

    if not results:
        #print("Warning: No results found in vector database")
        return []

    results_str = [doc.page_content for doc in results]
    results_pair = {doc.page_content: doc.metadata['tool_adaptation'] for doc in results}

    if len(results_str) <= topk:
        #print(f"Warning: Only {len(results_str)} chunks available, fewer than requested {topk}")
        return [results_pair[scene] for scene in results_str]

    try:
        # reranker
        rank, score = rerank(query, results_str, with_score=True, top_n=topk)
    except Exception as e:

        return [results_pair[scene] for scene in results_str[:topk]]

    topk_chunks = []
    for i, doc_idx in enumerate(rank):
        if i < topk and doc_idx < len(results_str):
            topk_chunks.append(results_pair[results_str[doc_idx]])

    return topk_chunks


# ========== Tool retrieval utils ==========
async def get_topk_similar_chunks(
    query: str,
    topk: int = 5,
    config: Config = None,
    node_llm: CachedChatOpenAI = None,
    used_hashes: Set[str] = set(),
) -> List[str]:
    """
    Retrieve the top-K text chunks that are semantically similar to the query.

    Args:
        query (str): Query string used to retrieve similar chunks.
        config (Config): Global configuration object.

    Returns:
        List[str]: List of top-K similar text chunks.
    """
    try:
        topk = int(topk)
    except (TypeError, ValueError):
        topk = 5

    # Clamp topk to safe range
    MIN_TOPK, MAX_TOPK = 2, 20
    topk = max(MIN_TOPK, min(topk, MAX_TOPK))

    # Initialize vector DB
    embedding_client = get_embedding(config)
    vector_store = Chroma(
        collection_name="Chunk",
        embedding_function=embedding_client,
        persist_directory=config.output_vectorDB_storage_path,
    )

    # First-stage retrieval
    retriever = vector_store.as_retriever(search_kwargs={"k": 200})
    results = retriever.invoke(query)

    # Filter out already-used chunks
    filtered_docs = [
        doc for doc in results if doc.metadata.get("hash") not in used_hashes
    ]
    candidate_texts = [doc.page_content for doc in filtered_docs]

    # Rerank
    rank, score = rerank(
        query, candidate_texts, with_score=True, top_n=topk
    )

    # Collect final top-K chunks and mark hashes as used
    topk_chunks = []
    for idx in rank:
        topk_chunks.append(candidate_texts[idx])
        used_hashes.add(filtered_docs[idx].metadata.get("hash"))

    return topk_chunks


async def get_chunks_by_exact_match(
    query: str,
    config: Config,
    node_llm: CachedChatOpenAI,
    used_hashes: Set[str] = set(),
    min_match_length: int = 2,
    max_semantic_candidates: int = 100,
) -> List[str]:
    if not query or not query.strip():
        return []

    query = query.strip()
    query_lower = query.lower()

    embedding_client = get_embedding(config)
    vector_store = Chroma(
        collection_name="Chunk",
        embedding_function=embedding_client,
        persist_directory=config.output_vectorDB_storage_path,
    )

    word_count = len(query.split())

    # Strategy 1: Short queries - direct full-text search
    if word_count <= 5:
        return await _direct_keyword_search(
            query, query_lower, vector_store, used_hashes, node_llm
        )

    # Strategy 2: Medium-long queries - semantic retrieval first, then exact matching
    else:
        # Get semantic retrieval candidates
        semantic_k = min(max_semantic_candidates, 200)
        retriever = vector_store.as_retriever(search_kwargs={"k": semantic_k})
        semantic_results = retriever.invoke(query)

        # Filter out already used documents
        filtered_docs = [
            doc for doc in semantic_results
            if doc.metadata.get("hash") not in used_hashes
        ]

        if not filtered_docs:
            return []

        # Exact matching filtering
        matched_docs = []

        # Extract key phrases
        key_phrases = _extract_key_phrases(query)

        for doc in filtered_docs:
            content = doc.page_content
            content_lower = content.lower()

            # Multi-matching strategy
            match_score = _calculate_exact_match_score(
                query_lower, content_lower, key_phrases, min_match_length
            )

            if match_score > 0:
                matched_docs.append({
                    'doc': doc,
                    'content': content,
                    'match_score': match_score,
                    'hash': doc.metadata.get("hash")
                })

        if not matched_docs:
            return []

        # Sort by matching score
        matched_docs.sort(key=lambda x: x['match_score'], reverse=True)

        # Use rerank for further optimization
        top_k = min(5, len(matched_docs))
        candidate_contents = [item['content'] for item in matched_docs[:top_k * 2]]

        try:
            from .retrieving_utils import rerank
            rank_indices, scores = rerank(
                query, candidate_contents, with_score=True, top_n=top_k
            )

            final_results = []
            for idx in rank_indices:
                final_results.append(candidate_contents[idx])
                used_hashes.add(matched_docs[idx]['hash'])

            return final_results
        except:
            # Rerank failed, directly return the highest matching scores
            final_results = []
            for item in matched_docs[:top_k]:
                final_results.append(item['content'])
                used_hashes.add(item['hash'])
            return final_results


async def get_chunks_by_fuzzy_match(
    keyword: str,
    config: Config,
    node_llm: CachedChatOpenAI,
    used_hashes: Set[str] = set(),
    similarity_threshold: float = 0.2,
    max_candidates: int = 150,
    top_k: int = 5,
) -> List[str]:
    if not keyword or not keyword.strip():
        return []

    keyword = keyword.strip().lower()
    word_count = len(keyword.split())
    char_count = len(keyword)

    # Initialize vector storage
    embedding_client = get_embedding(config)
    vector_store = Chroma(
        collection_name="Chunk",
        embedding_function=embedding_client,
        persist_directory=config.output_vectorDB_storage_path,
    )

    candidates = []

    # Strategy 1: Super short keywords (1-2 words) - direct search
    if word_count <= 2 or char_count <= 15:
        candidates = await _direct_fuzzy_search(
            keyword, vector_store, used_hashes, max_candidates
        )

    # Strategy 2: Medium length keywords - hybrid strategy
    elif word_count <= 10:
        # Semantic retrieval first to get candidates, then fuzzy matching
        semantic_k = min(max_candidates // 2, 100)
        retriever = vector_store.as_retriever(search_kwargs={"k": semantic_k})
        semantic_results = retriever.invoke(keyword)

        filtered_docs = [
            doc for doc in semantic_results
            if doc.metadata.get("hash") not in used_hashes
        ]

        # Add some random documents for diversity
        try:
            additional_k = min(max_candidates - len(filtered_docs), 50)
            if additional_k > 0:
                import random
                all_retriever = vector_store.as_retriever(search_kwargs={"k": 500})
                all_results = all_retriever.invoke("random diverse content")
                additional_docs = [
                    doc for doc in all_results
                    if doc.metadata.get("hash") not in used_hashes and
                    doc not in filtered_docs
                ][:additional_k]
                filtered_docs.extend(additional_docs)
        except:
            pass

        candidates = filtered_docs

    # Strategy 3: Long queries - primarily semantic retrieval
    else:
        semantic_k = min(max_candidates, 200)
        retriever = vector_store.as_retriever(search_kwargs={"k": semantic_k})
        semantic_results = retriever.invoke(keyword)

        candidates = [
            doc for doc in semantic_results
            if doc.metadata.get("hash") not in used_hashes
        ]

    if not candidates:
        return []

    # Calculate fuzzy matching scores
    scored_docs = []
    for doc in candidates:
        fuzzy_score = _calculate_fuzzy_similarity_fast(keyword, doc.page_content)

        if fuzzy_score >= similarity_threshold:
            scored_docs.append({
                'doc': doc,
                'content': doc.page_content,
                'fuzzy_score': fuzzy_score,
                'hash': doc.metadata.get("hash")
            })

    if not scored_docs:
        return []

    # Sort by scores
    scored_docs.sort(key=lambda x: x['fuzzy_score'], reverse=True)

    # Final selection
    final_results = []
    selected_count = min(top_k, len(scored_docs))

    for i in range(selected_count):
        final_results.append(scored_docs[i]['content'])
        used_hashes.add(scored_docs[i]['hash'])

    return final_results


# Helper functions

async def _direct_keyword_search(
    query: str,
    query_lower: str,
    vector_store: Chroma,
    used_hashes: Set[str],
    node_llm: CachedChatOpenAI
) -> List[str]:
    """Direct keyword search, suitable for short queries"""
    # Get a large number of candidate documents for full-text search
    retriever = vector_store.as_retriever(search_kwargs={"k": 500})
    all_results = retriever.invoke("general content")  # Use general query to get more documents

    # Filter out already used documents
    filtered_docs = [
        doc for doc in all_results
        if doc.metadata.get("hash") not in used_hashes
    ]

    # Exact matching
    matched_docs = []
    key_phrases = _extract_key_phrases(query)

    for doc in filtered_docs:
        content_lower = doc.page_content.lower()

        # Check complete query match
        if query_lower in content_lower:
            matched_docs.append({
                'doc': doc,
                'content': doc.page_content,
                'match_score': 1.0,
                'hash': doc.metadata.get("hash")
            })
            continue

        # Check key phrase matches
        phrase_matches = sum(1 for phrase in key_phrases if phrase.lower() in content_lower)
        if phrase_matches > 0:
            matched_docs.append({
                'doc': doc,
                'content': doc.page_content,
                'match_score': phrase_matches / len(key_phrases),
                'hash': doc.metadata.get("hash")
            })

    # Sort by score and return top results
    matched_docs.sort(key=lambda x: x['match_score'], reverse=True)

    results = []
    for item in matched_docs[:5]:
        results.append(item['content'])
        used_hashes.add(item['hash'])

    return results


async def _direct_fuzzy_search(
    keyword: str,
    vector_store: Chroma,
    used_hashes: Set[str],
    max_candidates: int
) -> List[Document]:
    """Direct fuzzy search, suitable for short keywords"""
    # Get candidate documents
    retriever = vector_store.as_retriever(search_kwargs={"k": max_candidates})
    results = retriever.invoke("diverse content")

    # Filter out already used documents
    filtered_docs = [
        doc for doc in results
        if doc.metadata.get("hash") not in used_hashes
    ]

    return filtered_docs


def _extract_key_phrases(query: str) -> List[str]:
    """Extract key phrases from the query"""
    # Simple key phrase extraction strategy
    phrases = []

    # Remove punctuation and split
    cleaned = re.sub(r'[^\w\s]', ' ', query)
    words = cleaned.split()

    # Add complete query
    if len(words) > 0:
        phrases.append(' '.join(words))

    # Add 2-gram phrases
    if len(words) >= 2:
        for i in range(len(words) - 1):
            phrases.append(f"{words[i]} {words[i+1]}")

    # Add important words (words with length >= 3)
    important_words = [w for w in words if len(w) >= 3]
    phrases.extend(important_words)

    return list(set(phrases))  # Remove duplicates


def _calculate_exact_match_score(
    query_lower: str,
    content_lower: str,
    key_phrases: List[str],
    min_match_length: int
) -> float:
    """Calculate exact matching score"""
    score = 0.0

    # Complete query match (highest weight)
    if query_lower in content_lower:
        score += 1.0
        return score  # Direct return highest score for complete match

    # Key phrase matching
    phrase_matches = 0
    for phrase in key_phrases:
        if len(phrase) >= min_match_length and phrase.lower() in content_lower:
            phrase_matches += 1
            score += 0.3

    # Word matching
    query_words = set(query_lower.split())
    content_words = set(content_lower.split())

    word_matches = len(query_words & content_words)
    if word_matches > 0:
        score += word_matches / len(query_words) * 0.5

    return score


def _calculate_fuzzy_similarity_fast(query: str, text: str) -> float:
    """Fast fuzzy similarity calculation"""
    if not query or not text:
        return 0.0

    query_lower = query.lower()
    text_lower = text.lower()

    # Fast pre-filtering
    if query_lower not in text_lower and len(set(query_lower) & set(text_lower)) < 2:
        return 0.0

    scores = []

    # 1. Substring matching score (weight 0.4)
    if query_lower in text_lower:
        substring_score = 1.0
    else:
        # Longest common substring
        lcs_len = _longest_common_substring_length(query_lower, text_lower)
        substring_score = lcs_len / len(query_lower) if len(query_lower) > 0 else 0.0

    scores.append(substring_score * 0.4)

    # 2. Jaccard similarity (weight 0.3)
    query_tokens = set(re.findall(r'\w+', query_lower))
    text_tokens = set(re.findall(r'\w+', text_lower))

    if query_tokens and text_tokens:
        jaccard = len(query_tokens & text_tokens) / len(query_tokens | text_tokens)
        scores.append(jaccard * 0.3)
    else:
        scores.append(0.0)

    # 3. Character-level similarity (weight 0.3)
    # For long texts, only calculate the first 500 characters
    text_sample = text_lower[:500] if len(text_lower) > 500 else text_lower
    char_similarity = SequenceMatcher(None, query_lower, text_sample).ratio()
    scores.append(char_similarity * 0.3)

    return sum(scores)


def _longest_common_substring_length(s1: str, s2: str) -> int:
    """Calculate longest common substring length (optimized version)"""
    if len(s1) < len(s2):
        s1, s2 = s2, s1  # Ensure s1 is the longer string

    max_len = 0
    current = [0] * (len(s2) + 1)

    for i in range(1, len(s1) + 1):
        previous = current.copy()
        for j in range(1, len(s2) + 1):
            if s1[i-1] == s2[j-1]:
                current[j] = previous[j-1] + 1
                max_len = max(max_len, current[j])
            else:
                current[j] = 0

    return max_len
from argparse import ArgumentParser
from elasticsearch import Elasticsearch, helpers
import html
import json
import os
from tqdm import tqdm
from typing import List, Dict, Any

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "../../.."))
from config import Config


INDEX_NAME = "hotpotqa_test"


def load_hotpotqa_data(input_dir: str) -> List[Dict[str, Any]]:
    """加载 HotpotQA 数据集并转换为可索引格式"""
    file_path = os.path.join(input_dir, "hotpot_test_200.json")
    with open(file_path, encoding="utf-8") as f:
        dev_dataset = json.load(f)

    documents = []
    doc_id = 0
    for data in dev_dataset:
        for title, paragraphs in data["context"]:
            text = " ".join(paragraphs)
            documents.append({
                "id": doc_id,
                "title": title,
                "text": text
            })
            doc_id += 1
    return documents


def create_index_mapping() -> Dict[str, Any]:
    """创建 Elasticsearch 索引结构"""
    return {
        "settings": {
            "number_of_shards": 1,
            "number_of_replicas": 0,
            "analysis": {
                "analyzer": {
                    "my_english_analyzer": {
                        "type": "standard",
                        "stopwords": "_english_",
                    },
                    "bigram_analyzer": {
                        "tokenizer": "standard",
                        "filter": ["lowercase", "stop", "shingle", "asciifolding"],
                    },
                }
            },
        },
        "mappings": {
            "properties": {
                "id": {"type": "integer"},
                "title": {"type": "text", "analyzer": "my_english_analyzer"},
                "text": {"type": "text", "analyzer": "my_english_analyzer"},
                "original_json": {"type": "text"},
            }
        },
    }


def generate_actions(docs: List[Dict[str, Any]]):
    """生成 bulk 索引 actions"""
    for doc in docs:
        yield {
            "_index": INDEX_NAME,
            "_id": f"wiki-{doc['id']}",
            "_source": {
                "id": doc["id"],
                "title": html.unescape(doc["title"]),
                "text": doc["text"],
                "original_json": json.dumps(doc, ensure_ascii=False)
            },
        }


def main(args):
    config = Config(dataset="hotpotqa_test")
    input_dir = config.input_corpus_dir

    print(f"📂 Loading HotpotQA_test data from: {input_dir}")
    documents = load_hotpotqa_data(input_dir)
    print(f"✅ Loaded {len(documents)} documents")

    # 初始化 Elasticsearch 客户端
    es = Elasticsearch(
        hosts=["http://localhost:9200"],
        timeout=60,
        max_retries=3,
        retry_on_timeout=True,
    )

    # 检查连接
    if not es.ping():
        raise RuntimeError("❌ Cannot connect to Elasticsearch at http://localhost:9200")

    # 创建索引
    if es.indices.exists(index=INDEX_NAME):
        if args.reindex:
            print(f"♻️ Deleting existing index: {INDEX_NAME}")
            es.indices.delete(index=INDEX_NAME)
        else:
            print(f"⚠️ Index '{INDEX_NAME}' already exists, skipping creation.")
    else:
        print(f"🧱 Creating index: {INDEX_NAME}")
        es.indices.create(index=INDEX_NAME, body=create_index_mapping())

    # 批量导入文档
    if not args.dry:
        print(f"🚀 Indexing documents into '{INDEX_NAME}' ...")
        success, failed = 0, 0
        for ok, result in helpers.streaming_bulk(
            client=es,
            actions=generate_actions(documents),
            chunk_size=200,  # ✅ 批次
            max_retries=3,
        ):
            if ok:
                success += 1
            else:
                failed += 1

        print(f"✅ Indexed {success} documents")
    else:
        print("🧩 Dry run, no indexing performed.")


if __name__ == "__main__":
    parser = ArgumentParser(description="Build Elasticsearch sparse index for HotpotQA_test")
    parser.add_argument("--reindex", action="store_true", help="Force reindex (delete existing index)")
    parser.add_argument("--dry", action="store_true", help="Dry run without indexing, for test")
    args = parser.parse_args()

    main(args)

import sys
import os
import csv, json
import time
import asyncio
from tqdm import tqdm
import datetime
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.agent import create_tca_agent
from config import Config
from src.logger import setup_logger
import argparse

parser = argparse.ArgumentParser()

parser.add_argument(
    "--dataset",
    type=str,
    default="hotpotqa",
    choices=["test", "hotpotqa", "hotpotqa_test", "2wiki", "2wiki_test", "musique", "musique_test"],
    help="Dataset to use",
)

parser.add_argument(
    "--mode",
    type=str,
    default="TEST",
    choices=["TEST", "ACCUMULATE"],
    help="Mode to use"
)

args = parser.parse_args()
config = Config(dataset=args.dataset)

timestamp = config.timestamp

# logger = setup_logger(f"acet_rag_{config.dataset}", "log/acet_rag")

def load_qas_evidences():

    if args.dataset == "hotpotqa":
        with open(f'../input/corpus/hotpotqa/hotpot_dev_200.json', 'r', encoding='utf-8') as f:
            dataset = json.load(f)

        queries = []
        answers = []
        evidences_list = []

        for item in dataset:
            queries.append(item['question'])
            answers.append(item['answer'])
            contexts = {}
            for c in item['context']:
                contexts[c[0]] = c[1]
            evidences = []
            for fact in item['supporting_facts']:
                evidences.append(contexts[fact[0]][fact[1]])
            evidences_list.append(evidences)

        return queries, answers, evidences_list

    elif args.dataset == "hotpotqa_test":
        with open(f'../input/corpus/hotpotqa_test/hotpot_test_200.json', 'r', encoding='utf-8') as f:
            dataset = json.load(f)

        queries = []
        answers = []
        evidences_list = []

        for item in dataset:
            queries.append(item['question'])
            answers.append(item['answer'])
            contexts = {}
            for c in item['context']:
                contexts[c[0]] = c[1]
            evidences = []
            for fact in item['supporting_facts']:
                evidences.append(contexts[fact[0]][fact[1]])
            evidences_list.append(evidences)

        return queries, answers, evidences_list

    elif args.dataset == "2wiki":
        with open(f'../input/corpus/2wiki/2wiki_dev_200.json', 'r', encoding='utf-8') as f:
            dataset = json.load(f)

        queries = []
        answers = []
        evidences_list = []

        for item in dataset:
            queries.append(item['question'])
            answers.append(item['answer'])
            contexts = {}
            for c in item['context']:
                contexts[c[0]] = c[1]
            evidences = []
            for fact in item['supporting_facts']:
                evidences.append(contexts[fact[0]][fact[1]])
            evidences_list.append(evidences)

        return queries, answers, evidences_list

    elif args.dataset == "2wiki_test":
        with open(f'../input/corpus/2wiki_test/2wiki_test_200.json', 'r', encoding='utf-8') as f:
            dataset = json.load(f)

        queries = []
        answers = []
        evidences_list = []

        for item in dataset:
            queries.append(item['question'])
            answers.append(item['answer'])
            contexts = {}
            for c in item['context']:
                contexts[c[0]] = c[1]
            evidences = []
            for fact in item['supporting_facts']:
                evidences.append(contexts[fact[0]][fact[1]])
            evidences_list.append(evidences)

        return queries, answers, evidences_list

    elif args.dataset == "musique":
        with open(f'../input/corpus/musique/musique_dev_200.json', 'r', encoding='utf-8') as f:
            dataset = json.load(f)

        queries = []
        answers = []
        evidences_list = []

        for item in dataset:
            queries.append(item['question'])
            answers.append(item['answer'])
            evidences = []
            for para in item['paragraphs']:
                if para['is_supporting']:
                    evidences.append(para['paragraph_text'])
            evidences_list.append(evidences)

        return queries, answers, evidences_list

    elif args.dataset == "musique_test":
        with open(f'../input/corpus/musique_test/musique_test_200.json', 'r', encoding='utf-8') as f:
            dataset = json.load(f)

        queries = []
        answers = []
        evidences_list = []

        for item in dataset:
            queries.append(item['question'])
            answers.append(item['answer'])
            evidences = []
            for para in item['paragraphs']:
                if para['is_supporting']:
                    evidences.append(para['paragraph_text'])
            evidences_list.append(evidences)

        return queries, answers, evidences_list



async def main():

    agent = create_tca_agent(config)

    queries, answers, evidences_list = load_qas_evidences()

    queries = queries
    answers = answers
    evidences_list = evidences_list

    results = []

    for query, answer, evidences in tqdm(zip(queries, answers, evidences_list), total=len(queries), desc="Quering"):
        start_time = time.time()
        if args.mode == "ACCUMULATE":
            agent_result = await agent.invoke(query, ground_truth=answer, evidences=evidences, mode="ACCUMULATE")
        else:
            agent_result = await agent.invoke(query, mode="TEST")
        end_time = time.time()
        total_time = end_time - start_time
        results.append(
            {
                "query": query,
                "result": agent_result,
                "answer": answer,
                "time": total_time,
                "number of tool calls": len(agent.action_history)
            }
        )

        os.makedirs(f"../result/acet_rag/{args.dataset}", exist_ok=True)
        with open(f"../result/acet_rag/{args.dataset}/{args.dataset}_result_{timestamp}.json", 'w') as f:
            json.dump(results, f, indent=4)

    agent.save_adaptation()

if __name__ == "__main__":
    asyncio.run(main())
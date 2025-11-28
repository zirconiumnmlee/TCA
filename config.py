import os, json
import datetime

class Config:
    def __init__(self, dataset=None, project_dir=None):

        self.dataset = dataset
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.timestamp = timestamp

        # log config
        self.log_path = f"log/acet/{timestamp}"
        self.log_name = f"{dataset}"

        # parallel log config
        self.parallel_log_path = f"log/acet/{timestamp}"
        self.parallel_log_names = [f"{dataset}_parallel_{i}" for i in range(4)]

        # ========= 1. Working Path Configuration =========
        if project_dir:
            self.project_dir = project_dir
        else:
            self.project_dir = os.path.dirname(__file__)

        # ========= 2. Input Output Paths =========
        self.input_corpus_dir = os.path.join(
            self.project_dir, f"input/corpus/{dataset}"
        )
        self.input_eval_dir = os.path.join(
            self.project_dir, f"input/eval/{dataset}"
        )

        self.output_dir = os.path.join(self.project_dir, f"output/{dataset}")
        self.work_dir = self.output_dir
        os.makedirs(self.output_dir, exist_ok=True)

        self.output_vectorDB_storage_path = os.path.join(self.output_dir, "chroma_db")
        self.output_chunk_json_path = os.path.join(
            self.output_dir, "kv_store_text_chunks.json"
        )


        # ========= 3. Record config ==========
        self.record_path = os.path.join(
            os.path.join(self.output_dir, f"records/{dataset}_record_{timestamp}.json")
        )
        os.makedirs(os.path.join(self.output_dir,"records"), exist_ok=True)
        with open(self.record_path, 'w') as f:
            json.dump([], f, indent=4)


        self.tool_memory_path = os.path.join(
            self.project_dir, f"memory/tool_memory/{dataset}_tool_memory_{timestamp}.json"
        )

        self.trajectory_memory_path = os.path.join(
            self.project_dir, f"memory/trajectory_memory/{dataset}_trajectory_memory_{timestamp}.json"
        )

        self.trajectory_memory_vectorDB_storage_path = os.path.join(
            self.project_dir, f"memory/trajectory_memory/{dataset}_trajectory_memory_{timestamp}_chroma_db"
        )


        # ========= 4. Vector Database Configuration =========
        self.vectorDB_embedding_dim = 384


        # ========= 5. Large Language Model Configuration =========
        self.single_llm_config = os.path.join(self.project_dir, "config/base.yaml")



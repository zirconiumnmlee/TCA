import yaml
from langchain_openai import OpenAIEmbeddings
from langchain_ollama import OllamaEmbeddings
from config import Config


def get_embedding(config: Config):
    """
    Get the embedding configuration based on the provided YAML config file.

    Args:
        config_path (str): Path to the YAML configuration file.

    Returns:
        object: An instance of the embedding class based on the configuration.
    """
    with open(config.single_llm_config, "r", encoding="utf-8") as file:
        base_config = yaml.safe_load(file)
    embedding_name = base_config["embedding"]["use"]
    embedding_config = base_config["embedding"].get(embedding_name)

    if embedding_name is None:
        raise ValueError("Invalid embedding use in config")

    if embedding_name == "nomic-embed-text":
        return OllamaEmbeddings(
            model=embedding_config["model"],
            temperature=embedding_config.get("temperature"),
            base_url=embedding_config["api_base"],
            dimensions=config.vectorDB_embedding_dim,
        )
    elif "model" in embedding_config:
        return OpenAIEmbeddings(
            model=embedding_config["model"],
            openai_api_key=embedding_config.get("api_key"),
            openai_api_base=embedding_config["api_base"],
            dimensions=config.vectorDB_embedding_dim,
            check_embedding_ctx_length=False,
        )
    else:
        return OpenAIEmbeddings(
            openai_api_key=embedding_config.get("api_key"),
            openai_api_base=embedding_config["api_base"],
            dimensions=config.vectorDB_embedding_dim,
        )

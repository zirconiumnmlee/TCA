import asyncio
import os
import yaml
from pathlib import Path
from langchain_openai import ChatOpenAI
from dataclasses import dataclass
from tenacity import retry, stop_after_attempt

from config import Config

project_dir = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)


@dataclass
class PackedResponse:
    content: str

class CachedChatOpenAI:
    def __init__(self, base_llm):
        self.llm = base_llm
        self.cache_dir = Path(f"{project_dir}/.cache/llm_responses")
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def invoke(self, *args, **kwargs):
        result = self.llm.invoke(*args, **kwargs)

        if hasattr(result, "content"):
            content = result.content
        elif isinstance(result, str):
            content = result
        else:
            content = str(result)

        return content

    async def ainvoke(self, *args, **kwargs):

        if hasattr(self.llm, 'ainvoke'):
            result = await self.llm.ainvoke(*args, **kwargs)
        else:
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(None, self.llm.invoke, *args, **kwargs)

        if hasattr(result, "content"):
            content = result.content
        elif isinstance(result, str):
            content = result
        else:
            content = str(result)

        return content


def get_llm(config: Config):
    """
    Get the language model with randomized OpenAI API key.

    Args:
        config_path (str): path to the config file

    Returns:
        the language model with caching capability
    """
    with open(config.single_llm_config, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    llm_name = config["llm"]["use"]
    llm_config = config["llm"].get(llm_name)

    if llm_name is None:
        raise ValueError("invalid llm use in config")
    else:
        base_llm = ChatOpenAI(
            model=llm_config["model"],
            openai_api_key=llm_config["api_key"],
            openai_api_base=llm_config["api_base"],
            temperature=llm_config.get("temperature"),
            max_tokens=llm_config.get("max_tokens"),
            timeout=llm_config.get("timeout"),
        )
    ### 待实现：在config中指明是否要缓存
    return CachedChatOpenAI(base_llm)


# @retry(stop=stop_after_attempt(5))
# def parse_llm_response(llm_client, prompt, logger: logging.Logger):
#     try:
#         response = llm_client.invoke(prompt)

#         if response.startswith("```json") and response.endswith("```"):
#             response = response[7:-3].strip()
#         response = response.replace("\n", " ").strip()

#         # parse json
#         return json.loads(response)

#     except json.JSONDecodeError as e:
#         logger.error(
#             f"❌❌❌ Retry {parse_llm_response.retry.statistics['attempt_number']} times, JSON parse failed: {str(e)}"
#         )
#         raise ValueError(f"JSON parse failed: {str(e)}") from e

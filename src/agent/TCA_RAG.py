import json
from unicodedata import category
import os
import sys
import datetime
import time
from typing import Any, Dict, Optional, Tuple, List

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(project_root)

from config import Config
from src.llm import CachedChatOpenAI, get_llm
from src.tools.base import ToolsRegistry
from src.logger import setup_logger
from src.prompt import PROMPTS
from src.memory import Memory
from src.utils.retrieving_utils import get_topk_tool_experience


class TCA_RAG:
    def __init__(self, config: Config = None, enabled_tools: list = None, logger=None):

        self.config = config

        # logger config
        if logger == None:
            self.logger = setup_logger(self.config.log_name, self.config.log_path)
        else:
            self.logger = logger

        # LLM config
        self.llm = get_llm(self.config)

        # tool config
        if enabled_tools:
            ToolsRegistry.set_enabled_tools(enabled_tools)

        self.available_tools = ToolsRegistry.get_enabled_tools()
        self.tool_instances = {}
        self._initialize_tools()

        # iteration config
        self.max_iterations = 5

        # history
        self.thought_history = []
        self.thought_full = []
        self.action_history = []
        self.observation_history = []

        # retrieval
        self.used_hashes = set()  # Avoid duplicate retrieval

        # memory module
        self.memory = Memory(config.tool_memory_path, config.trajectory_memory_path, logger=self.logger, llm=self.llm)
        self.logger.info(f"Set Tools Adaptation at {self.memory.tool_memory_path}(ToolMemory), {self.memory.trajectory_memory_path}(TrajectoryMemory)")

        # Compare
        self.current_tool_preference = None

        self.loop_tool_call_stat = {}

        # mode control
        self.mode = "ACCUMULAE"

        # parallel configuration
        self.tool_layer_adaptation = []
        self.trajectory_layer_adaptation = []
        self.parallel_num = len(self.available_tools)
        self.parallel_loggers = []

    # Main loop method
    async def loop(self, query: str, ground_truth: str=None, evidences: List[str]=None) -> str:
        """
        Do ReAct loop: Think-Act-Observe
        Args:
            query: user's query
        Returns:
            final answer
        """
        self.logger.info(f"Start ReAct loop: {query}")

        # Reset history for new loop
        self.used_hashes = set()
        self.thought_history = []
        self.action_history = []
        self.observation_history = []
        self.loop_tool_call_stat = {}

        for iteration in range(self.max_iterations):
            self.logger.info(f"\n{'='*50}")
            self.logger.info(f"Iteration {iteration + 1}/{self.max_iterations}")
            self.logger.info(f"{'='*50}")

            action_plan = await self._think(query, iteration)
            self.thought_history.append(action_plan["thought"])

            if action_plan["action"] == "answer":
                self.logger.info("Query can be answered, ReAct loop finished")
                break

            action_result = await self._act(action_plan, query_context=query)
            self.action_history.append({
                "action": action_plan["action"],
                "input": action_plan.get("action_input"),
                "result": action_result
            })

            observation = self._observe(action_result, action_plan)
            self.observation_history.append(observation)


            if isinstance(action_result, dict) and "error" in action_result:
                self.logger.warning(f"Tool execution error, continue to the next round: {action_result['error']}")

        # Maximum number of iterations reached, force answer generation
        if iteration == self.max_iterations:
            self.logger.warning(f"Reaches the maximum number of iterations {self.max_iterations}")

        final_answer = await self._generate_final_answer(query)

        # Tool-layer experience accumulation
        if self.mode == "ACCUMULAE":
            await self._tool_accumulate(self.action_history, evidences)
        if self.mode == "TEST":
            loop_record = self._save_loop_record_TEST(query, final_answer)
            try:
                with open(self.config.record_path, 'r') as f:
                    records = json.load(f)
            except (FileNotFoundError, json.JSONDecodeError):
                records = []

            records.append(loop_record)
            with open(self.config.record_path, 'w') as f:
                json.dump(records, f, indent=4)

        return final_answer

    async def multi_loop(self, query: str, ground_truth: str, evidences:List[str]=None, tools_list: List[str]=None) -> List[str]:

        tool_preferences = self._get_tool_preferences_list(tools_list)
        n_compare = len(tool_preferences)
        self.logger.info(f"Starting serial comparison learning mode, total {n_compare} rounds")

        trajectories = []
        final_answers = []
        loop_record = {
            "query": query,
            "ground_truth": ground_truth,
            "multi_loop":[]
        }

        # Execute each round serially
        for i in range(n_compare):
            preference = tool_preferences[i]

            self.logger.info(f"\n{'='*60}")
            self.logger.info(f"Comparison learning round {i+1}/{n_compare} (serial), tool preference: {preference}")
            self.logger.info(f"{'='*60}")

            try:
                # Execute react loop serially
                final_answer = await self.loop(query, ground_truth, evidences)
                trajectory = self._build_trajectory(final_answer, ground_truth, preference)

                self.logger.info(f"Round {i+1} completed, answer: {final_answer[:100]}...")

                record = self._save_loop_record(query, final_answer, ground_truth, preference)

                # Write to data structure (consistent with original parallel version)
                final_answers.append(final_answer)
                trajectories.append(trajectory)
                loop_record["multi_loop"].append(record)

            except Exception as e:
                self.logger.error(f"Round {i+1} execution failed: {e}")
                final_answers.append(f"Execution failed: {str(e)}")
                trajectories.append(f"Execution trajectory: {str(e)}")

        # Clear tool preference
        self.current_tool_preference = None

        # Save records (maintaining original logic completely)
        try:
            with open(self.config.record_path, 'r') as f:
                records = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            records = []
        records.append(loop_record)
        with open(self.config.record_path, 'w') as f:
            json.dump(records, f, indent=4)

        # Comparison learning
        self.logger.info(f"\n{'='*60}")
        self.logger.info("Starting comparison learning for all trajectories (serial results)...")
        self.logger.info(f"{'='*60}")

        await self._compare_accumulate(query, trajectories)

        return final_answers


    async def _think(self, query: str) -> Dict[str, Any]:
        """
        Think Stage: Analyze the situation and determine a complete action plan
        """
        try:
            thought_full = {
                "pre-think":"",
                "trajectory-level":"",
                "tool-level":"",
                "post-think":""
            }
            context = self._build_context_for_think()

            # === 1. Pre-Think phase ===
            pre_prompt = PROMPTS['pre_think'].format(
                query=query,
                context=context,
                tool_descriptions=self.get_tool_descriptions(),
            )

            pre_think_response = await self.llm.ainvoke(pre_prompt)
            pre_think_result = self.safe_json_parse(pre_think_response.strip())

            if not pre_think_result or "scene" not in pre_think_result or "candidate_tools" not in pre_think_result:
                self.logger.error(f"Pre-think failed to parse, response: {pre_think_response}")
                pre_think_result = {
                    "scene": "Pre-think failed, proceed conservatively.",
                    "candidate_tools": ["retrieve_default"],
                }

            thought_full['pre-think'] = pre_think_result

            pre_think_summary = (
                f"scene: {pre_think_result['scene']}\n"
                f"candidate_tools: {pre_think_result['candidate_tools']}\n"
            )

            self.logger.info(f"Pre-think summary: {pre_think_summary}")

            # === 2. Retrieve experience ===
            trajectory_level_tool_adaptations = self.get_trajectory_layer_experience_for_think(pre_think_result['scene'])
            tool_level_tool_adaptations = self.get_tool_layer_experience_for_think(pre_think_result['candidate_tools'])
            tool_adaptations = {
                "trajectory_level_tool_adaptations": trajectory_level_tool_adaptations,
                "tool_level_tool_adaptations": tool_level_tool_adaptations
            }

            thought_full['trajectory-level'] = trajectory_level_tool_adaptations
            thought_full['tool-level'] = tool_level_tool_adaptations

            # === 3. Post-Think phase ===
            # If there is a tool preference, add it to the tool adaptation information
            preference_info = f"Current Tool Preference: \n{self.current_tool_preference}" if self.current_tool_preference else ""

            post_prompt = PROMPTS['post_think'].format(
                query=query,
                context=context,
                pre_think=pre_think_summary,
                tool_descriptions=self.get_tool_descriptions(),
                tool_preference=preference_info,
                trajectory_level_tool_adaptations=tool_adaptations["trajectory_level_tool_adaptations"],
                tool_level_tool_adaptations=tool_adaptations["tool_level_tool_adaptations"],
            )

            post_response = await self.llm.ainvoke(post_prompt)
            post_result = self.safe_json_parse(post_response.strip())

            if not post_result or "thought" not in post_result or "action" not in post_result:
                self.logger.error(f"Post-think failed to parse, response: {post_response}")
                post_result = {
                    "thought": "Post-think failed, fallback to direct answer.",
                    "action": "answer",
                    "action_input": None
                }

            thought_full['post-think'] = post_result
            self.thought_full.append(thought_full)

            self.logger.info(f"Think: {post_result['thought']}")
            self.logger.info(f"Action decision: {post_result['action']}")

            return post_result

        except Exception as e:
            self.logger.error(f"Think stage failed: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return {
                "thought": f"Error in think stage: {str(e)}",
                "action": "answer",
                "action_input": None
            }

    async def _act(self, action_plan: Dict[str, Any], query_context: str = "") -> Any:
        """
        Act stage: The act of carrying out a decision made during the thinking stage
        Args:
            action_plan: A dict containing action and action_input
            query_context: Context of the user query
        Returns:
            act result
        """
        action = action_plan["action"]
        action_input = action_plan.get("action_input") or {}

        try:
            # Execute tool
            if action in self.tool_instances:
                self.logger.info(f"Performing action: using tool {action}")
                self.loop_tool_call_stat[action] = self.loop_tool_call_stat.get(action, 0) + 1
                result = await self.execute_tool(action, query_context=query_context, **action_input)
                return result
            else:
                error_msg = f"Unknown tool: {action}"
                self.logger.error(error_msg)
                return {"error": error_msg}

        except Exception as e:
            self.logger.error(f"Act stage failed: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return {"error": f"Perform action failed: {str(e)}"}

    def _observe(self, action_result: Any, action_plan: Dict[str, Any]) -> str:
        """
        Observe stage: Analyze the results of actions and form observation reports
        Args:
            action_result: Results of implementation of actions
            action_plan: Original action plan
        Returns:
            string of observation report
        """
        action = action_plan["action"]

        try:
            # Check for errors
            if isinstance(action_result, dict) and "error" in action_result:
                observation = f"❌ Tool '{action}' execute failed: {action_result['error']} Failure reason：{action_result['answer']}"

            else:
                # Tool executed successfully
                if isinstance(action_result, dict):
                    observation = f"✓ Tool '{action}' execution finished: {action_result}"
                else:
                    observation = f"✓ Tool '{action}' execution finished: {str(action_result)[:200]}..."

            self.logger.info(f"Observe: {observation}")
            return observation

        except Exception as e:
            self.logger.error(f"Observe stage failed: {e}")
            return f"observation process failed: {str(e)}"

    async def _compare_accumulate(self, query: str, trajectories: List[str] ):
        # 1. Organize multiple trajectories
        multi_trajectories = []
        id = 1
        for trajectory in trajectories:
            multi_trajectories.append(f"Trajectory:{id}")
            id+=1
            multi_trajectories.append(trajectory)
        multi_trajectories = "\n".join(multi_trajectories)

        # 2. Generate prompt
        prompt = PROMPTS['compare_adapt'].format(
            query = query,
            trajectories = multi_trajectories
        )
        self.logger.info("LLM starting to learn from multiple trajectories...")
        response = await self.llm.ainvoke(prompt)

        # 3. Parse response to get tool adaptation
        parsed_response = self.safe_json_parse(response)
        scene = parsed_response['scene']
        tool_adaptation = parsed_response['tool_adaptation']

        self.trajectory_layer_adaptation.append(
            {
                "query": query,
                "multi_trajectories": multi_trajectories,
                "scene": scene,
                "tool_adaptation": tool_adaptation
            }
        )

        self.logger.info(f"Successfully stored adaptation for query: {query[:100]}...")

    async def _tool_accumulate(self, action_history:Dict, evidences:List[int]):
        self.logger.info("LLM starting to learn from tool calls...")
        for action in action_history:
            tool_name = action['action']
            tool_input = action['input']
            chunks = action['result']['answer']
            recall = self.cal_recall(chunks, evidences)
            self.logger.info(f"Recall: {recall}")

            if recall == 0:
                category = "POOR"
            elif recall < 0.8:
                category = "PARTIAL"
            else:
                category = "EXCELLENT"

            prompt = PROMPTS['tool_adapt'][category].format(
                tool_name = tool_name,
                tool_input = tool_input,
                tool_output = chunks,
                evidences = evidences,
                recall = recall
            )
            response = await self.llm.ainvoke(prompt)
            parsed_response = self.safe_json_parse(response)

            tool_adaptation = parsed_response['tool_adaptation']

            self.tool_layer_adaptation.append(
                {
                    "tool_name": tool_name,
                    "category": category,
                    "tool_adaptation": tool_adaptation
                }
            )

    def save_adaptation(self):

        # with open("./trajectory_layer_adaptation.json", "w") as f:
        #     json.dump(self.trajectory_layer_adaptation, f, indent=4)
        # with open("./tool_layer_adaptation.json", "w") as f:
        #     json.dump(self.tool_layer_adaptation, f, indent=4)

        self.logger.info(">"*30+"Start save adaptation")
        # trajectory layer
        self.logger.info("Start save trajectory-layer adaptation...")
        for traj_layer_adapt in self.trajectory_layer_adaptation:
            query = traj_layer_adapt['query']
            multi_trajectories = traj_layer_adapt['multi_trajectories']
            scene = traj_layer_adapt['scene']
            tool_adaptation = traj_layer_adapt['tool_adaptation']
            # Store trajectory, query, tool_adaptation to JSON file
            self.memory.add_trajectory_memory(query, multi_trajectories, scene, tool_adaptation)
            # Create vector embedding for tool_adaptation and store in vector database
            self._store_trajectory_memory_vector(query, scene, tool_adaptation)
        self.logger.info(f"Save {len(self.trajectory_layer_adaptation)} trajectory-layer adaptation")

        # tool layer
        self.logger.info("Start save tool-layer-adaptation...")
        for tool_layer_adapt_one_query in self.tool_layer_adaptation:
            for tool_layer_adapt in tool_layer_adapt_one_query:
                tool_name = tool_layer_adapt['tool_name']
                category = tool_layer_adapt['category']
                tool_adaptation = tool_layer_adapt['tool_adaptation']
                self.memory.add_tool_memory(tool_name, category, tool_adaptation)
        self.logger.info(f"Save {len(self.tool_layer_adaptation)} tool-layer adaptation")

        self.logger.info("<"*30+"Finish save adaptation")

    # system output 方法
    async def _generate_final_answer(self, query: str = "") -> str:
        """
        Generate final answer
        Args:
            query: user's query
        Returns:
            final answer
        """
        try:
            context = self._build_context_for_final_answer()

            # Extract all successful retrieval results
            retrieval_results = []
            for action in self.action_history:
                result = action.get("result", {})
                if isinstance(result, dict) and "answer" in result and "error" not in result:
                    retrieval_results.extend(result["answer"])

            retrieval_context = "\n\n".join(retrieval_results) if retrieval_results else "No relevant information was retrieved"

            prompt = PROMPTS['generate_final_answer'].format(
                query=query,
                retrieval_context=retrieval_context,
                context=context
            )

            final_answer = await self.llm.ainvoke(prompt)
            self.logger.info(f"Final answer has been generated (length: {len(final_answer)})")
            self.logger.info(f"Final answer: {final_answer}\n========================= FINISH =========================\n")

            return final_answer.strip()

        except Exception as e:
            self.logger.error(f"Failed to generate the final answer: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return f"Sorry, an error was encountered while generating the answer: {str(e)}"


    ## Agent interface methods
    async def invoke(self, query: str, ground_truth: str=None, evidences: List[str]=None, mode: str="TEST", tool_preferences: List[str]=None) -> str:
        """
        Main entry point: Execute ReAct loop with support for comparative learning

        Args:
            query: User query
            ground_truth: Ground truth answer (for learning and evaluation)
            mode: Running mode, "ACCUMULAE" for adaptive learning mode, "TEST" for testing mode
            tool_preferences: Tool preference list, if None, defaults to all tools

        Returns:
            Final answer
        """
        if mode not in {"ACCUMULAE", "TEST"}:
            raise ValueError(f"Invalid mode: {mode}. Mode must be 'ACCUMULAE' or 'TEST'")

        self.mode = mode

        if self.mode == "TEST":
            # Testing mode: Standard single execution
            return await self.loop(query)

        elif self.mode == "ACCUMULAE":
            # Adaptive learning mode: Enable comparative learning
            if ground_truth is None:
                raise ValueError("ground_truth is required in ACCUMULAE mode")

            if evidences is None:
                raise ValueError("evidences is required in ACCUMULAE mode")

            return await self.multi_loop(query, ground_truth, evidences, tool_preferences)


    ## Context building methods
    def _build_context_for_think(self) -> str:
        """
        Build current context information
        Returns:
            Formatted context string
        """
        context_parts = []

        # Historical thoughts
        if self.thought_history:
            context_parts.append("Thought history:")
            for i, thought in enumerate(self.thought_history[-3:], 1):
                context_parts.append(f"  {i}. {thought}")

        # Historical actions
        if self.action_history:
            context_parts.append("\nAction history:")
            for i, action in enumerate(self.action_history[-3:], 1):
                action_name = action.get("action", "unknown")
                context_parts.append(f"  {i}. {action_name}")

        # Historical observations
        if self.observation_history:
            context_parts.append("\nObservation history:")
            for i, observation in enumerate(self.observation_history[-3:], 1):
                context_parts.append(f"  {i}. {observation}")

        if not context_parts:
            return "This is the 1st iteration, so there is no history information"

        return "\n".join(context_parts)

    def _build_context_for_final_answer(self) -> str:
        """Build complete historical context (for final answer generation)"""
        context_parts = []

        for i in range(len(self.thought_history)):
            context_parts.append(f"\n--- Iteration {i+1} ---")

            if i < len(self.thought_history):
                context_parts.append(f"Think: {self.thought_history[i][:200]}...")

            if i < len(self.action_history):
                action = self.action_history[i]
                context_parts.append(f"Action: {action.get('action', 'unknown')}")

            if i < len(self.observation_history):
                context_parts.append(f"Observation: {self.observation_history[i][:200]}...")

        return "\n".join(context_parts)

    ## Record saving methods
    def _save_loop_record(self, query: str, final_answer: str, ground_truth: str, tool_preference: str = None) -> dict:
        """
        Generate dictionary record for a single trajectory, used for comparative learning
        Args:
            query: User query
            final_answer: Final answer
            ground_truth: Ground truth answer
            tool_preference: Tool preference
        Returns:
            Dictionary containing complete trajectory information
        """
        trajectory_text = self._build_trajectory(final_answer, ground_truth, tool_preference)

        return {
            "tool_preference": tool_preference or "No preference",
            "tool_call_stat": self.loop_tool_call_stat,
            "trajectory": trajectory_text,
            "final_answer": final_answer,
            "thought_history": self.thought_history.copy(),
            "action_history": self.action_history.copy(),
            "observation_history": self.observation_history.copy()
        }

    def _save_loop_record_TEST(self, query: str, final_answer: str) -> dict:

        return {
            "final_answer": final_answer,
            "tool_call_stat": self.loop_tool_call_stat,
            "thought_full": self.thought_full,
            "thought_history": self.thought_history.copy(),
            "action_history": self.action_history.copy(),
            "observation_history": self.observation_history.copy()
        }


    ## Experience usage methods
    def get_trajectory_layer_experience_for_think(self, query:str, topk:int=5) -> str:
        tool_adaptation_list = get_topk_tool_experience(query, config=self.config, topk=topk)
        self.logger.info(f"Trajectory-level tool Adaptations: get {len(tool_adaptation_list)} adaptations.")
        trajectory_level_tool_adapatations = "\n".join(tool_adaptation_list)
        return trajectory_level_tool_adapatations

    def get_tool_layer_experience_for_think(self, candidate_tools:List[str]) -> str:
        tool_adapatations = []
        num_adapt = 0
        with open(self.config.tool_memory_path, 'r') as f:
            tool_memory = json.load(f)

        for candidate in candidate_tools:
            tool_adapatations.append(f"Tool {candidate} using handbook:")
            if candidate in tool_memory:
                tool_adapatations.append(f"Learned from Failure:")
                for adaptation in tool_memory[candidate]["POOR"][:min(len(tool_memory[candidate]["POOR"]), 3)]:
                    tool_adapatations.append(f"- {adaptation['tool_adaptation']}")
                    num_adapt += 1

                tool_adapatations.append(f"Learned from Partial:")
                for adaptation in tool_memory[candidate]["PARTIAL"][:min(len(tool_memory[candidate]["PARTIAL"]), 3)]:
                    tool_adapatations.append(f"- {adaptation['tool_adaptation']}")
                    num_adapt += 1

                tool_adapatations.append(f"Learned from Success:")
                for adaptation in tool_memory[candidate]["EXCELLENT"][:min(len(tool_memory[candidate]["EXCELLENT"]), 3)]:
                    tool_adapatations.append(f"- {adaptation['tool_adaptation']}")
                    num_adapt += 1
            else:
                tool_adapatations.append("- No previous experiences found.")

        self.logger.info(f"Tool-level tool Adaptations: get {num_adapt} adaptations.")
        tool_level_tool_adapatations = "\n".join(tool_adapatations)
        return tool_level_tool_adapatations


    def _get_tool_preferences_list(self, tools_list:List[str]=None) -> List[str]:
        if tools_list == None:
            tools_list = list(self.available_tools)

        preferences = []

        for tool_name in tools_list:
            preferences.append(f"Prefer to use {tool_name} tool.")

        return preferences

    # trajectory layer
    def _build_trajectory(self, final_answer, ground_truth, tool_preference=None)->str:
        """
        Generate complete trajectory record for a single react loop
        Args:
            final_answer: Final answer
            ground_truth: Ground truth answer
            tool_preference: Tool preference used in this execution
        Returns:
            Complete trajectory string
        """
        trajectory = []

        # Add tool preference information
        if tool_preference:
            trajectory.append(f"Tool Preference: {tool_preference}")

        for i in range(len(self.action_history)):
            action = self.action_history[i]
            input_paras = ""
            for input_name, input_content in action['input'].items():
                input_paras += f"{input_name}={input_content},"
            action_desc = f"{action['action']}({input_paras})"

            trajectory.append(
                f"\nThought: {self.thought_history[i]}\nAction: {action_desc}\nObservation: {self.observation_history[i]}"
            )

        if len(self.thought_history)>len(self.action_history):
            trajectory.append(
                f"\nThought: {self.thought_history[-1]}\nAction: Answer the question now."
            )
        else:
            trajectory.append("\nReached the maximum number of iterations.")

        trajectory.append(f"\nFinal answer: {final_answer}")
        trajectory.append(f"\nGround truth: {ground_truth}")
        trajectory = "".join(trajectory)
        return trajectory

    def _store_trajectory_memory_vector(self, query: str, scene: str, tool_adaptation: str):
        """
        Store tool_adaptation as vector embedding in vector database

        Args:
            query: Original query
            tool_adaptation: Tools adaptation content
        """
        try:
            from langchain_core.documents import Document
            from langchain_chroma import Chroma
            from ..llm.embedding import get_embedding

            # Create document content
            document = Document(
                page_content=scene,
                metadata={
                    "query": query,
                    "scene": scene,
                    "tool_adaptation": tool_adaptation,
                    "timestamp": datetime.datetime.now().isoformat(),
                },
            )

            # Initialize embedding client and vector store
            embedding_client = get_embedding(self.config)
            vector_store = Chroma(
                collection_name="TrajectoryMemory",
                embedding_function=embedding_client,
                persist_directory=self.config.trajectory_memory_vectorDB_storage_path,
            )

            # Add document to vector store
            vector_store.add_documents([document])

            self.logger.info("Successfully stored tool_adaptation in vector database")

        except Exception as e:
            self.logger.error(f"Failed to store tool_adaptation in vector database: {e}")
            import traceback
            self.logger.error(traceback.format_exc())


    ## Tool-related methods
    def _initialize_tools(self):
        """Initializes all enabled tool instances"""
        for tool_name in self.available_tools:
            tool_class = ToolsRegistry.get_tools([tool_name])[tool_name]
            try:
                self.tool_instances[tool_name] = tool_class(self)
                self.logger.info(f"Successfully initialized tool: {tool_name}")
            except Exception as e:
                self.logger.error(f"Failed to initialize tool {tool_name}: {e}")

    def set_tool_preference(self, preference: str):
        """
        Set current tool preference
        Args:
            preference: Tool name
        """
        self.current_tool_preference = preference
        self.logger.info(f"Set tool preference: {preference}")

    def get_tool_descriptions(self) -> str:
        """Get all enabled tool descriptions"""
        descriptions = []
        for tool_name, tool_class in ToolsRegistry.get_tools().items():
            descriptions.append(f"- {tool_name}: {tool_class.get_tool_description()}")
        return "\n".join(descriptions)

    async def execute_tool(self, tool_name: str, query_context: str = "", **kwargs) -> Dict[str, Any]:
        """
        Execute designated tool
        Args:
            tool_name: Name of tool
            query_context: Context of the user query that triggered this tool call
            **kwargs: Dictionary parameters

        Returns:
            Result of tool execution
        """
        if tool_name not in self.tool_instances:
            return {"error": f"Tool '{tool_name}' is not available"}

        # Record execution start time
        start_time = time.time()
        success = False
        result = None

        try:
            tool_instance = self.tool_instances[tool_name]
            kwargs['node_llm'] = self.llm

            self.logger.info(f"Tool: {tool_name}, Parameters: {kwargs}")
            result = await tool_instance.execute(**kwargs)

            self.logger.info(f"Tool {tool_name} execution finished")
            success = True
            return result

        except Exception as e:
            self.logger.error(f"Tool failed to execute: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            result = {"error": f"Tool failed to execute: {str(e)}"}
            success = False
            return result

        # finally:
        #     # Record execution in memory
        #     try:
        #         execution_time = time.time() - start_time
        #         # Define keys that don't need serialization
        #         NON_SERIALIZABLE_KEYS = {'node_llm', 'llm', 'agent', 'logger', 'config'}

        #         # Filter input_data
        #         serializable_input = {
        #             k: v for k, v in kwargs.items()
        #             if k not in NON_SERIALIZABLE_KEYS
        #         }

        #         # Filter output_data (if it's a dictionary)
        #         if isinstance(result, dict):
        #             serializable_output = {
        #                 k: v for k, v in result.items()
        #                 if k not in NON_SERIALIZABLE_KEYS
        #             }
        #         else:
        #             serializable_output = result

        #         self.memory.add_tool_record(
        #             tool_name=tool_name,
        #             input_data=serializable_input,
        #             output_data=serializable_output,
        #             success=success,
        #             query_context=query_context
        #         )
        #         self.logger.debug(f"Tool execution recorded in memory: {tool_name} ({execution_time:.3f}s)")
        #     except Exception as e:
        #         self.logger.warning(f"Failed to record tool execution in memory: {e}")


    # Helper methods
    def safe_json_parse(self, text):
        """
        Safely parse JSON output from LLM, automatically fix common errors.
        """
        import re, json

        if not text or not isinstance(text, str):
            return None

        raw = text.strip()

        # Step 1: Remove Markdown wrapper symbols
        raw = re.sub(r"```(?:json)?", "", raw).strip("` \n\t\r")

        # Step 2: Extract JSON body
        start = min([p for p in [raw.find("{"), raw.find("[")] if p != -1], default=0)
        end = max([p for p in [raw.rfind("}"), raw.rfind("]")] if p != -1], default=len(raw))
        raw = raw[start:end + 1]

        # Step 3: Try parsing directly
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            pass

        fixed = raw

        # Step 4: Common fixes (conservative version)
        fixed = re.sub(r"[\x00-\x1f\x7f-\x9f]", "", fixed)  # Remove control characters
        fixed = fixed.replace("“", '"').replace("”", '"').replace("’", "'")
        fixed = re.sub(r",\s*([}\]])", r"\1", fixed)        # Remove trailing commas

        # Single-quote pseudo-JSON → Double-quote JSON
        if fixed.strip().startswith("{") and "'" in fixed and '"' not in fixed:
            fixed = fixed.replace("'", '"')

        # Python style values → JSON
        fixed = fixed.replace("None", "null").replace("True", "true").replace("False", "false")

        # Balance brackets
        if fixed.count("{") > fixed.count("}"):
            fixed += "}" * (fixed.count("{") - fixed.count("}"))
        if fixed.count("[") > fixed.count("]"):
            fixed += "]" * (fixed.count("[") - fixed.count("]"))

        # Step 5: Try parsing
        try:
            return json.loads(fixed)
        except Exception as e:
            self.logger.error(f"Still failed to parse JSON: {e}")
            self.logger.error(f"Original: {text}")
            self.logger.error(f"Cleaned: {fixed}")
            return None


    def cal_recall(self, chunks: List, evidences: List):
        if evidences == [] or chunks == []:
            self.logger.info("chunks or evidences is empty.")
            return 0
        num_evidences = len(evidences)
        retrieved_content = "".join(chunks)
        found = 0
        for e in evidences:
            if e in retrieved_content:
                found+=1
        recall = found/num_evidences
        return recall


# Function to create agent
def create_acet_agent(config: Config = None, enabled_tools: list = None, logger=None):
    """
    Factory function to create Agent instance

    Args:
        config: Configuration object
        enabled_tools: List of enabled tools
        logger: Logger instance

    Returns:
        Agent instance
    """
    return TCA_RAG(config, enabled_tools, logger)



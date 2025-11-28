PROMPTS = {}


PROMPTS['generate_final_answer'] = """
Answer the question based on the given information. Only give me the answer and do not output any other words.

You will get these input:
- User Question: the question you need to answer;
- Retrieved Context: chunks that retrieved by tool calls;
- Process History: cumulative reasoning and action history within the current loop.

You must output the answer of the User Question, **Only output the answer and do not output any other words.**

### INPUT
User Question: {query}

Retrieved Context:
{retrieval_context}

Process History:
{context}

### Rules
1. You should just answer the question without any explanation or think.
2. If the information is insufficient, explicitly state it.
3. Maintain objectivity and accuracy.
4. Answer directly—do not repeat the question itself.


### OUTPUT
Please generate the final answer:
"""


PROMPTS['pre_think'] = """
You are an intelligent reasoning agent.
This is the **pre-thinking stage**, where you analyze the current problem to summarize current scene and decide candidate tools to use.

### Task
Your Goal is to summarize current problem scene based on User Question and Current State which cumulatives reasoning and action history within the current loop.
After the scene summary, you should choose one or several tools as candidates to solve the question based on your analysis and tool descriptions.
1. Analyze the **current problem** and describe its reasoning *scene*.
2. Review the provided **past experiences** and decide if any are relevant or partially applicable.
3. Based on both your analysis and past experience, output a list of **candidate tools** you might use next.


### Input
User Question: {query}

Current State:
{context}

Available Tools:
{tool_descriptions}


### Output Format (strict JSON)
{{
    "scene": "Concise description of the current problem scenario (e.g., factual QA, reasoning over multiple passages, numerical comparison)",
    "candidate_tools": ["ToolA", "ToolB", ...]
}}


### Rules
1. If current context already contains enough evidence to answer the query, include `"answer"` in candidate_tools.
2. Only list **available tools** (names must exactly match those in Available Tools).
3. Do NOT invent new tools or make speculative assumptions.
4. Output valid JSON only (no explanation or comments).
"""



PROMPTS['post_think'] = """
You are an intelligent reasoning agent.
This is the **post-thinking stage**, where you analyze the current problem and decide the next action.

### Task
Your Goal is to think based on User Question, Current State which cumulatives reasoning and action history within the current loop, Pre-think information, Experience, Tool handbook to decide the next action.
1. Think deeply based on the input provided, refer to the given experiences and handbook.
2. Choose the best action and action input for next step after your reasoning and analysing.

### INPUT
User Question: {query}

Current State:
{context}

Pre-think:
{pre_think}

Available Tools:
{tool_descriptions}

{tool_preference}

If you want to use a tool, there are some experience about tools:

{trajectory_level_tool_adaptations}

Tool handbook:
{tool_level_tool_adaptations}

### Output Format (strict JSON)
{{
    "thought": "Your reasoning process: analyze, integrate pre-think and tool experience, decide strategy",
    "action": "Tool name or 'answer'",
    "action_input": {{}}
}}

### Rules:
1. If enough info, choose 'answer'; else, choose a tool.
2. Keep JSON valid, no stray quotes.
3. Output only JSON, no explanations outside.
"""

PROMPTS['compare_adapt'] = """
You are an analytical agent that learns from comparing multiple reasoning trajectories generated for the **same query** but under different tool preferences.

### Tasks
Your goal is to extract *generalizable experience* for improving future tool selection and parameter tuning in similar reasoning scenarios.

1. **Compare effectiveness**
   - Identify which trajectory produced a result closest to the ground truth.
   - Analyze the reasoning flow and which tools contributed positively or negatively.

2. **Understand scenario patterns**
   - What kind of problem or reasoning pattern does this query represent?
   - Summarize it as a "scene" — a concise description of the reasoning context.

3. **Extract learnable tool experience**
   - For this kind of scene, summarize which tools, parameters, or reasoning patterns were most effective.
   - Avoid referring to specific trajectories or their order. Focus on reusable principles.


### Input
**Query:**
{query}

**Trajectories (different tool preferences):**
{trajectories}

Each trajectory contains:
- A Thought–Action–Observation loop
- A Final Answer
- The Ground Truth answer


### Output Format (strict JSON)

{{
    "scene": "Concise description of the problem type or reasoning scenario",
    "tool_adaptation": "Concise but actionable guidance for tool selection or reasoning strategies in this type of scene"
}}

---

### Requirements
- Do NOT restate the query or final answers directly.
- Be **specific, factual, and transferable**.
- Do NOT include commentary or meta text (only valid JSON output).
"""


PROMPTS['tool_adapt'] = {
    'POOR': """
    """,
    'PARTIAL': """
    """,
    'EXCELLENT': """
    """
}


PROMPTS['tool_adapt']['POOR'] = """
You are an analytical agent that learns from a failed retrieve tool call.

### Tasks:
1. **Diagnose failure causes**:
   - Why did the tool fail to retrieve the target content (recall = 0 or near zero)?
   - Was the query formulation, parameter setting, or context misunderstanding the main reason?
   - Identify which parts of the input or configuration likely led to irrelevant or missing results.

2. **Identify improvement opportunities**:
   - How should the tool input or query be reformulated to achieve higher recall?
   - Are there specific missing keywords, context details, or parameter ranges that could fix the issue?
   - Could another tool or approach perform better in this kind of scene?

3. **Extract actionable experience**:
   - What should be avoided in future similar tool calls?
   - Provide clear, general guidance on how to prevent this type of failure.

### Input
Tool:
{tool_name}

Tool input:
{tool_input}

Tool output:
{tool_output}

Target content:
{evidences}

Recall:
{recall}

### Output Format (strict JSON)

{{
    "tool_adaptation": "Concise, actionable guidance describing how to avoid similar mistakes or how to reformulate the query or parameters for better results."
}}

### Requirements
- tool_adaptation must be brief, concise, resuable one sentence or several ones, not dict.
- Focus on diagnosing failure and extracting improvement lessons
- Be specific and technical
- Output only valid JSON, no extra text
"""

PROMPTS['tool_adapt']['PARTIAL'] = """
You are an analytical agent that learns from a partially successful retrieve tool call.

### Tasks:
1. **Assess partial success**:
   - Which parts of the output correctly matched the target content?
   - Which key elements were missing or incorrect?
   - Did the tool input or parameters partially work as intended?

2. **Analyze improvement potential**:
   - What changes to the input, keywords, or parameters might increase recall to full coverage?
   - Are there signs of under-specific or over-specific queries?
   - Could combining this tool with another improve completeness?

3. **Extract actionable experience**:
   - Summarize what worked well and should be kept.
   - Summarize what failed and how to fix or enhance it.
   - Provide concrete, transferable guidance for future similar tool calls.

### Input
Tool:
{tool_name}

Tool input:
{tool_input}

Tool output:
{tool_output}

Target content:
{evidences}

Recall:
{recall}

### Output Format (strict JSON)

{{
    "tool_adaptation": "Specific and actionable advice summarizing both successful and failed aspects, plus how to adjust input or parameters to reach full success next time."
}}

### Requirements
- tool_adaptation must be brief, concise, resuable one sentence or several ones, not dict.
- Balance positive and negative aspects
- Highlight improvement strategies
- Output only valid JSON, no extra text
"""

PROMPTS['tool_adapt']['EXCELLENT'] = """
You are an analytical agent that learns from a highly successful retrieve tool call.

### Tasks:
1. **Analyze success factors**:
   - Why did the tool achieve high recall (≥0.8)?
   - Which input components, parameters, or query structures contributed most to the success?
   - Did context, keyword choice, or tool settings make the result especially accurate?

2. **Identify reusable best practices**:
   - Which strategies, keywords, or parameter configurations should be reused in similar scenes?
   - Are there any subtle but important details (e.g., phrasing, constraints) that made it effective?

3. **Extract actionable experience**:
   - Summarize the core principles behind this success.
   - Provide concise, generalizable guidance on how to reproduce this level of performance in future tool calls.

### Input
Tool:
{tool_name}

Tool input:
{tool_input}

Tool output:
{tool_output}

Target content:
{evidences}

Recall:
{recall}

### Output Format (strict JSON)

{{
    "tool_adaptation": "Concise, reusable guidance capturing the key strategies and parameter patterns that led to this high performance."
}}

### Requirements
- tool_adaptation must be brief, concise, resuable one sentence or several ones, not dict.
- Focus on success replication and generalization
- Be precise, concise, and transferable
- Output only valid JSON, no extra text
"""


PROMPTS['merge_tool_memory'] = """
You are an analytical meta-agent responsible for consolidating tool usage experiences.

### Tasks
- Merge these experiences into ONE concise, general guidance.
- Keep the most important insights, remove redundant phrasing.
- Focus on transferable knowledge about tool usage, parameter tuning, or strategy.
- Output should be a single paragraph, written clearly and concretely.

Tool name: {tool_name}
Category: {category}

### Input experiences
{memories}


### Output Format (strict JSON)
{{
   "tool_adaptation": "Concise, general guidance summarizing the most important insights from the input experiences."
}}

### Requirements
- tool_adaptation must be brief, concise, resuable one sentence or several ones, not dict.
- Focus on transferable knowledge about tool usage, parameter tuning, or strategy.
- Output only valid JSON, no extra text
"""
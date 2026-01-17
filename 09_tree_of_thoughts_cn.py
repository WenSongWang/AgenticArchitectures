#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
# 📘 智能体架构 9：思维树（Tree-of-Thoughts）规划

这种模式将智能体的问题解决能力从线性思维链提升到多路径探索搜索。

与生成单一、顺序的推理路线不同，思维树智能体在问题的每个阶段都会生成多个候选"思路"或下一步。
然后它评估这些思路，修剪无效或没有前途的分支，并扩展最有希望的分支。
这创建了一个搜索树，智能体可以回溯、探索替代方案，并系统地导航复杂的问题空间。

为了演示这一点，我们将让我们的智能体解决一个简单直观的逻辑难题：**数字求和路径问题**。
这个问题要求从数字1开始，每次只能加1或乘以2，在5步内到达数字10。
这个例子虽然简单，但清晰地展示了思维树如何并行探索多个解决方案路径，并通过评估和修剪来高效找到最优解。

### 定义
**思维树（Tree-of-Thoughts，ToT）**是一种智能体推理框架，其中问题解决被建模为通过树的搜索。
智能体同时探索多个推理路径（分支）。在每个步骤，它生成潜在的下一步或"思路"，评估它们的可行性，并决定继续探索哪些路径，从而有效地修剪搜索空间。

### 高级工作流程

1.  **分解**：将问题分解为一系列步骤或思路。
2.  **思路生成**：对于问题的当前状态，智能体生成多个潜在的下一步或思路。这会在搜索树中创建分支。
3.  **状态评估**：每个新思路（导致新状态）由"评论家"或验证函数评估。该评估可以评估：
    *   **有效性**：此移动是否符合问题规则？
    *   **进展**：此移动是否让我们更接近解决方案？
    *   **启发式**：这条路径是否可能成功？
4.  **修剪与扩展**：无效或没有前途的分支被修剪。然后智能体从最有前途的活动分支继续，重复思路生成过程。
5.  **解决方案**：该过程继续，直到达到目标状态。解决方案是从根到目标的思路路径。

### 何时使用 / 应用
*   **逻辑谜题和数学问题**：具有明确规则和目标状态的问题，需要多步骤、非线性推理（如数独、过河谜题）。
*   **复杂规划**：当任务需要详细计划，其中操作顺序很重要且必须遵守约束时（例如，计划具有多段行程和预算约束的复杂旅行）。
*   **创意写作或代码生成**：在提交之前探索多个故事分支或实现策略。

### 优势和劣势
*   **优势**：
    *   **稳健性**：系统地探索问题空间，与单次通过方法相比，不太可能卡住或产生错误答案。
    *   **处理组合复杂性**：非常适合可能序列数量巨大的问题。
*   **劣势**：
    *   **计算量大**：与简单的思维链提示相比，需要显著更多的LLM调用和状态管理，因此速度更慢、成本更高。
    *   **需要良好的评估器**：搜索的有效性在很大程度上取决于状态评估逻辑的质量。
"""

# 📋 环境与设置
# 我们将安装所有必要的库，并配置我们的API密钥。

# !pip install -q -U langchain langgraph rich python-dotenv langchain_community langchain-openai

import os
import re
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
from collections import defaultdict

# Pydantic用于数据建模
from pydantic import BaseModel, Field

# OpenAI相关
from openai import OpenAI
from openai import RateLimitError, APIError

# LangChain组件
from langchain_core.prompts import ChatPromptTemplate

# LangGraph组件
from langgraph.graph import StateGraph, END
from typing_extensions import TypedDict

# 用于美化打印
from rich.console import Console
from rich.markdown import Markdown
from rich.tree import Tree

# 配置调试模式
DEBUG: bool = True

# 初始化日志
import logging
from rich.logging import RichHandler

logger = logging.getLogger("tree_of_thoughts")
handler = RichHandler(console=Console(), rich_tracebacks=True, markup=True)
handler.setFormatter(logging.Formatter("%(message)s"))
logger.handlers = [handler]
logger.propagate = False
logger.setLevel(logging.DEBUG if DEBUG else logging.INFO)

# ModelScope的OpenAI兼容接口适配器
class ModelScopeChat:
    """
    ModelScope 的 OpenAI 兼容接口适配器：
    - 提供 invoke(prompt) 基本调用
    - 提供 with_structured_output(PydanticModel) 的结构化输出包装
    - 支持API错误时自动切换到备选模型
    """
    def __init__(self, base_url: str, api_key: str, model: str, temperature: float = 0.2, extra_body: Optional[dict] = None):
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.model = model
        # 从环境变量获取备选模型ID
        self.fallback_model = os.environ.get("MODELSCOPE_MODEL_ID_R1")
        self.base_url = base_url
        self.temperature = temperature
        self.extra_body = extra_body or {}
        self.switched = False

    def invoke(self, prompt: str):
        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                stream=False,
                extra_body=self.extra_body,
            )
            # 非流式返回：choices[0].message.content
            return resp.choices[0].message.content
        except (RateLimitError, APIError) as e:
            if not self.switched and self.fallback_model:
                if DEBUG:
                    console.print(f"[bold yellow]⚠️ 模型 {self.model} 请求失败: {str(e)}，尝试切换到备选模型 {self.fallback_model}[/bold yellow]")
                # 切换到备选模型
                self.model = self.fallback_model
                self.switched = True
                # 重新尝试请求
                resp = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=self.temperature,
                    stream=False,
                    extra_body=self.extra_body,
                )
                return resp.choices[0].message.content
            else:
                # 如果没有备选模型或已经切换过，重新抛出异常
                raise

    def with_structured_output(self, pyd_model: type[BaseModel]):
        class _StructuredWrapper:
            def __init__(self, outer: "ModelScopeChat"):
                self.outer = outer

            def invoke(self, prompt: str) -> BaseModel:
                # 通过系统提示约束仅输出 JSON（尽量提高解析成功率），并明确字段/类型
                schema = pyd_model.model_json_schema()
                props = schema.get("properties", {})
                required = schema.get("required", [])
                schema_text_lines = []
                for k, v in props.items():
                    t = v.get("type", "string")
                    schema_text_lines.append(f"- {k}: {t}")
                schema_text = "\n".join(schema_text_lines) or "- 请按模型定义生成字段"
                required_text = ", ".join(required) if required else "所有字段"
                system_msg = (
                    "你是一个结构化输出生成器。只输出一个 JSON 对象，严格匹配以下字段与类型：\n"
                    f"{schema_text}\n"
                    f"必须包含字段：{required_text}\n"
                    "不要输出任何解释或多余文本（例如代码块标记、前后缀）。"
                )
                if DEBUG:
                    logger.debug("🔧 生成动态系统提示（包含字段与类型要求）")
                    logger.debug("结构化输出提示（系统消息）：\n" + system_msg)
                messages = [
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": prompt},
                ]
                try:
                    resp = self.outer.client.chat.completions.create(
                        model=self.outer.model,
                        messages=messages,
                        temperature=self.outer.temperature,
                        stream=False,
                        extra_body=self.outer.extra_body,
                    )
                except (RateLimitError, APIError) as e:
                    if not self.outer.switched and self.outer.fallback_model:
                        if DEBUG:
                            console.print(f"[bold yellow]⚠️ 模型 {self.outer.model} 请求失败: {str(e)}，尝试切换到备选模型 {self.outer.fallback_model}[/bold yellow]")
                        # 切换到备选模型
                        self.outer.model = self.outer.fallback_model
                        self.outer.switched = True
                        # 重新尝试请求
                        resp = self.outer.client.chat.completions.create(
                            model=self.outer.model,
                            messages=messages,
                            temperature=self.outer.temperature,
                            stream=False,
                            extra_body=self.outer.extra_body,
                        )
                    else:
                        # 如果没有备选模型或已经切换过，重新抛出异常
                        raise
                content = resp.choices[0].message.content or ""
                import json, re
                from pydantic import ValidationError
                def _extract_json(s: str) -> str:
                    m = re.search(r'(\{[\s\S]*\}|\[[\s\S]*\])', s)
                    return m.group(1) if m else "{}"
                raw = content.strip()
                if DEBUG:
                    console.print("[bold blue]📥 收到模型返回，尝试解析为 JSON[/bold blue]")
                try:
                    data = json.loads(raw)
                except Exception:
                    data = json.loads(_extract_json(raw))
                if DEBUG:
                    console.print("[bold cyan]模型原始返回（截断展示）[/bold cyan]:")
                    preview = json.dumps(data, ensure_ascii=False)[:400]
                    console.print(preview + ("..." if len(preview) == 400 else ""))
                # 兜底字段映射：尽量把常见别名映射到目标模型字段
                try:
                    parsed = pyd_model.model_validate(data)
                    if DEBUG:
                        console.print(f"[bold green]✅ 结构化解析成功[/bold green]：{pyd_model.__name__}")
                    return parsed
                except ValidationError:
                    if DEBUG:
                        console.print("[bold yellow]⚠️ 字段不匹配，尝试自动映射常见别名[/bold yellow]")
                    mappings_applied = []
                    # KnowledgeGraph: 常见返回 'relationships' 字段
                    if "relationships" not in data and "relations" in data:
                        data["relationships"] = data.pop("relations")
                        mappings_applied.append("relations → relationships")
                    if DEBUG and mappings_applied:
                        console.print("[bold cyan]已应用字段映射：[/bold cyan] " + ", ".join(mappings_applied))
                    parsed = pyd_model.model_validate(data)
                    if DEBUG:
                        console.print(f"[bold green]✅ 结构化解析成功[/bold green]：{pyd_model.__name__}")
                    return parsed

        return _StructuredWrapper(self)


def init_llm() -> ModelScopeChat:
    """
    初始化 ModelScope LLM（OpenAI 兼容接口）。
    - 可通过环境变量控制：
      MODELSCOPE_BASE_URL（默认：https://api-inference.modelscope.cn/v1）
      MODELSCOPE_API_KEY
      MODELSCOPE_MODEL_ID（默认：deepseek-ai/DeepSeek-V3.2）
    - 额外参数：enable_thinking 可选
    """
    base_url = os.environ.get("MODELSCOPE_BASE_URL", "https://api-inference.modelscope.cn/v1")
    api_key = os.environ.get("MODELSCOPE_API_KEY", "")
    model_id = os.environ.get("MODELSCOPE_MODEL_ID", "deepseek-ai/DeepSeek-V3.2")
    # 为避免“未信任的 chat template”错误，增加信任参数；并请求 JSON 输出格式
    extra = {
        "enable_thinking": True,
        "trust_request_chat_template": True,
        "response_format": {"type": "json_object"},
    }
    return ModelScopeChat(base_url=base_url, api_key=api_key, model=model_id, temperature=0.4, extra_body=extra)


# --- API密钥和追踪设置 ---
load_dotenv()

# 禁用LangSmith追踪（如果API密钥已过期）
os.environ["LANGCHAIN_TRACING_V2"] = "false"
os.environ["LANGCHAIN_PROJECT"] = "智能体架构 - 思维树（ModelScope）"

# 检查所需的环境变量
required_vars = ["MODELSCOPE_API_KEY", "LANGCHAIN_API_KEY"]
missing_vars = []
for var in required_vars:
    if var not in os.environ:
        missing_vars.append(var)

if missing_vars:
    print(f"警告: 以下环境变量未设置: {', '.join(missing_vars)}")

print("环境变量已加载，追踪已设置。")

# 初始化控制台
console = Console()

# 初始化LLM
llm = init_llm()

# --- 第1阶段：定义问题环境 ---
# 思维树系统需要在一个定义明确的环境中运行。对于我们的数字求和路径问题，
# 这意味着我们需要以编程方式定义：
# 
# 1.  **状态表示**：描述当前数字和已走路径的方式。
# 2.  **验证规则**：检查状态是否有效的函数（例如，步数限制）。
# 3.  **目标状态**：检查谜题是否已解决的方式（是否到达目标数字）。
# 4.  **可能的移动**：确定从给定状态的所有合法移动的函数（加1或乘2）。

# 问题配置参数（更复杂的版本：使用+1、*3、-2操作到达29）
CONFIG = {
    "START_NUMBER": 1,
    "TARGET_NUMBER": 29,
    "MAX_STEPS": 8,  # 增加步数限制，因为操作更复杂
    "MOVE_OPTIONS": [
        ("+1", lambda x: x + 1),
        ("×3", lambda x: x * 3),
        ("-2", lambda x: x - 2)  # 新添加的操作
    ]
}

class NumberPathState(BaseModel):
    """表示数字求和路径问题的状态。"""
    current_number: int = Field(default=CONFIG["START_NUMBER"], description="当前数字")
    path: List[int] = Field(default_factory=lambda: [CONFIG["START_NUMBER"]], description="已走路径")
    steps_taken: int = Field(default=0, description="已走步数")
    move_description: str = Field(default=f"初始状态：从{CONFIG['START_NUMBER']}开始。", description="移动描述")

    def is_valid(self) -> bool:
        """检查当前状态是否有效（步数是否在限制内）。"""
        return self.steps_taken <= CONFIG["MAX_STEPS"]

    def is_goal(self) -> bool:
        """检查是否已达到目标状态（是否到达目标数字）。"""
        return self.current_number == CONFIG["TARGET_NUMBER"]
    
    def __hash__(self):
        # 使状态可哈希，以检查访问过的状态
        return hash((self.current_number, self.steps_taken))
    
    def __eq__(self, other):
        return self.__hash__() == other.__hash__()

def get_possible_moves(state: NumberPathState) -> list[NumberPathState]:
    """从当前状态生成所有可能的有效下一个状态（根据配置的移动选项）。"""
    moves = []
    
    # 遍历所有配置的移动选项
    for move_symbol, move_func in CONFIG["MOVE_OPTIONS"]:
        new_state = state.model_copy(deep=True)
        new_state.current_number = move_func(new_state.current_number)
        new_state.path.append(new_state.current_number)
        new_state.steps_taken += 1
        new_state.move_description = f"步骤 {new_state.steps_taken}: {new_state.path[-2]} {move_symbol} = {new_state.current_number}"
        
        if new_state.is_valid():
            moves.append(new_state)
        
    return moves

print("数字求和路径环境定义成功。")

# --- 第2阶段：使用LangGraph实现思维树智能体 ---
# 现在我们将构建智能体本身。我们图的状态将跟踪思维树中所有活动路径（分支）。
# 节点将执行关键的思维树操作：
# 
# 1.  **扩展路径（思路生成器）**：一个基于LLM的节点，查看每个活动路径的最后状态，
#     并从有效可能性列表中提出一个有前途的下一步。
# 2.  **修剪路径（状态评估器）**：这个节点在生成后进行清理。它将移除任何
#     进入无效状态或循环（重新访问之前状态）的路径。
# 3.  **检查解决方案（目标检查）**：一个条件节点，检查是否有任何活动路径已达到目标状态。
#     如果是这样，它将终止循环。

# LLM选择移动的Pydantic模型
class MoveChoice(BaseModel):
    best_move_index: int = Field(description="从提供的可能移动列表中最佳移动的索引。")
    reasoning: str = Field(description="为什么这是最有前途的移动的简要推理。")

# LangGraph状态
class ToTState(TypedDict):
    problem_description: str
    # 每个路径是NumberPathState对象的列表
    active_paths: List[List[NumberPathState]]
    # 我们将在这里存储最终解决方案
    solution: Optional[List[NumberPathState]]

# 图节点

def initialize_search(state: ToTState) -> Dict[str, Any]:
    """设置搜索初始状态的节点。"""
    initial_number_state = NumberPathState()
    return {"active_paths": [[initial_number_state]]}


def expand_paths(state: ToTState) -> Dict[str, Any]:
    """'思路生成器'。用有前途的下一步扩展每个活动路径。"""
    console.print("--- 扩展路径 ---")
    new_paths = []
    choice_llm = llm.with_structured_output(MoveChoice)
    
    # 生成动态系统提示，使用配置参数
    move_symbols = ", ".join([symbol for symbol, _ in CONFIG["MOVE_OPTIONS"]])
    system_prompt = (
        "您是一位逻辑谜题专家。您的目标是解决数字求和路径问题："
        f"从{CONFIG['START_NUMBER']}开始，每次只能{move_symbols}，"
        f"在{CONFIG['MAX_STEPS']}步内到达{CONFIG['TARGET_NUMBER']}。"
        "分析当前路径并从提供的选项列表中选择最有前途的下一步。"
    )
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "问题: {problem}\n\n当前路径历史:\n{path_history}\n\n从最终状态，从以下列表中选择最佳下一步:\n{possible_moves}")
    ])
    
    for path in state['active_paths']:
        last_state = path[-1]
        possible_next_states = get_possible_moves(last_state)
        
        if not possible_next_states:
            continue  # 这条路径是死胡同
            
        path_history_str = " -> ".join([s.move_description for s in path])
        possible_moves_str = "\n".join([f"{i}: {s.move_description}" for i, s in enumerate(possible_next_states)])
        
        # 为了简单和展示广度，我们可以探索多个移动。
        # 更高级的思维树可能会使用LLM只选择单个最佳移动。
        # 在这里，我们将让所有有效移动分支以展示树结构。
        for next_state in possible_next_states:
            new_paths.append(path + [next_state])

    console.print(f"[cyan]扩展到 {len(new_paths)} 条潜在路径。[/cyan]")
    return {"active_paths": new_paths}


def prune_paths(state: ToTState) -> Dict[str, Any]:
    """'状态评估器'。修剪无效或包含循环的路径，并基于启发式评估优先选择有希望的路径。"""
    console.print("--- 修剪路径 ---")
    valid_paths = []
    
    # 第一步：移除无效路径和循环
    for path in state['active_paths']:
        # 检查循环：如果最后一个状态之前在路径中出现过
        if path[-1] in path[:-1]:
            continue  # 发现循环，修剪这条路径
        
        # 检查有效性
        if path[-1].is_valid():
            valid_paths.append(path)
    
    # 第二步：基于启发式评估对路径进行排序
    def heuristic(path):
        last_state = path[-1]
        distance_to_goal = abs(CONFIG["TARGET_NUMBER"] - last_state.current_number)  # 距离目标的距离
        steps_efficiency = CONFIG["MAX_STEPS"] - last_state.steps_taken  # 剩余可用步数
        
        # 综合得分：距离越近得分越高，剩余步数越多得分越高
        # 添加一个惩罚项，避免数字过大（超过目标太多）
        overshoot_penalty = max(0, last_state.current_number - (CONFIG["TARGET_NUMBER"] * 2)) * 2
        return - (distance_to_goal - steps_efficiency + overshoot_penalty)
    
    # 按启发式得分排序
    valid_paths.sort(key=heuristic, reverse=True)
    
    # 第三步：只保留前10条最有希望的路径（避免搜索空间过大）
    pruned_paths = valid_paths[:10]
    
    console.print(f"[green]修剪后剩下 {len(pruned_paths)} 条有效、非循环且有希望的路径。[/green]")
    return {"active_paths": pruned_paths}


# 条件节点
def check_for_solution(state: ToTState) -> str:
    """检查是否有任何路径已达到目标并路由执行。"""
    for path in state['active_paths']:
        if path[-1].is_goal():
            console.print("[bold green]找到解决方案！[/bold green]")
            return "solution_found"
    return "continue_search"

# 添加一个新的节点来处理解决方案
def process_solution(state: ToTState) -> Dict[str, Any]:
    """找到并保存解决方案。"""
    for path in state['active_paths']:
        if path[-1].is_goal():
            return {
                "active_paths": state['active_paths'],
                "problem_description": state['problem_description'],
                "solution": path
            }
    return state

# 构建图
workflow = StateGraph(ToTState)

workflow.add_node("initialize", initialize_search)
workflow.add_node("expand", expand_paths)
workflow.add_node("prune", prune_paths)
workflow.add_node("process_solution", process_solution)  # 添加新的处理解决方案节点

workflow.set_entry_point("initialize")
workflow.add_edge("initialize", "expand")
workflow.add_edge("expand", "prune")

workflow.add_conditional_edges(
    "prune",
    check_for_solution,
    {
        "solution_found": "process_solution",  # 先处理解决方案
        "continue_search": "expand"
    }
)

workflow.add_edge("process_solution", END)  # 从处理节点到结束

tot_agent = workflow.compile()
print("思维树智能体图编译成功。")

# --- 第3阶段：演示与分析 ---
# 现在，让我们在谜题上运行我们的思维树智能体。我们将比较它的系统方法与简单的单次思维链请求，
# 以突出稳健性的差异。

problem_description = """
数字求和路径问题（升级版）：
从数字1开始，每次可以执行以下操作之一：+1、×3、-2
在8步内到达数字29。
请找出一条有效路径。
"""

# 运行思维树智能体
console.print("\n--- 🚀 运行思维树智能体解决谜题 ---")
result = tot_agent.invoke({
    "problem_description": problem_description,
    "active_paths": [],
    "solution": None
})

# 展示解决方案
if result.get("solution"):
    console.print("\n--- 📋 解决方案路径 ---")
    solution_path = result["solution"]
    
    # 创建可视化路径树
    path_tree = Tree("[bold blue]📈 解决方案路径[/bold blue]")
    for i, state in enumerate(solution_path):
        node_label = f"[{i+1}] {state.move_description}"
        node = path_tree.add(node_label)
        node.add(f"当前数字: {state.current_number} | 已走步数: {state.steps_taken}")
    
    console.print(path_tree)
    
    # 显示解决方案统计
    console.print("\n--- 📊 解决方案统计 ---")
    console.print(f"[green]✅ 总步数:[/green] {solution_path[-1].steps_taken}")
    console.print(f"[green]✅ 路径序列:[/green] {' → '.join(map(str, solution_path[-1].path))}")
    console.print(f"[green]✅ 解决效率:[/green] {(1 - (solution_path[-1].steps_taken / 5)) * 100:.1f}% (5步限制)")
    console.print()
else:
    console.print("[red]❌ 未找到解决方案。[/red]")

# 与简单思维链进行比较
console.print("\n--- 🧠 与简单思维链比较 ---")
simple_prompt = ChatPromptTemplate.from_messages([
    ("system", "您是一位逻辑谜题专家。请解决以下数字求和路径问题。"),
    ("human", "{problem}")
])

formatted_prompt = simple_prompt.format_messages(problem=problem_description)
formatted_prompt_str = formatted_prompt[0].content + "\n\n" + formatted_prompt[1].content

console.print("[yellow]运行简单思维链请求...[/yellow]")
chain_of_thought_response = llm.invoke(formatted_prompt_str)

console.print("\n--- 💬 思维链响应 ---")
console.print(chain_of_thought_response)

# 🔍 分析比较
console.print("\n--- 📊 分析比较 ---")
console.print("[green]思维树方法的优势：[/green]")
console.print("- 系统地探索问题空间，减少遗漏关键步骤的风险")
console.print("- 能够回溯并从死胡同中恢复")
console.print("- 明确验证每个状态的有效性")
console.print("- 可视化思维过程，便于调试和理解")

console.print("\n[red]思维链方法的局限性：[/red]")
console.print("- 线性思考可能会陷入局部最优解")
console.print("- 没有明确的机制来验证中间步骤")
console.print("- 难以回溯和修正错误决策")

# 📝 结论
# 在这个文件中，我们成功构建了一个使用思维树架构的问题解决智能体。
# 演示清楚地展示了这种方法的强大功能：
# 
# - **简单思维链的局限性**：单一的线性思考过程很容易忽略关键步骤或陷入无效状态。
# - **思维树的稳健性**：通过系统地探索多个路径并验证每个状态，思维树智能体能够找到正确的解决方案。
# - **可扩展性**：这种架构可以应用于更复杂的问题，如数学证明、代码生成和战略规划。
# 
# 虽然思维树方法需要更多的计算资源和LLM调用，但对于需要可靠性和系统性的复杂问题，这种额外的成本是值得的。

console.print("\n--- 🎉 思维树规划演示完成！ ---")

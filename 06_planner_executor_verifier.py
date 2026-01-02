# -*- coding: utf-8 -*-
"""
规划→执行→验证（Planner→Executor→Verifier）架构的可运行示例

学习目标（看完你能做到什么）：
- 理解PEV架构如何通过"规划→执行→验证"三步工作流实现错误检测与自修正
- 掌握LangGraph1.0中条件边（conditional edges）的使用，实现根据验证结果动态调整工作流
- 学会设计并实现一个验证器（Verifier）节点，能够检测执行结果的错误
- 能把脚本作为命令行程序运行，并观察PEV架构如何从错误中恢复

核心概念速览：
- PEV架构：Planner（规划）→ Executor（执行）→ Verifier（验证）的三步工作流
- 条件边：根据Verifier的验证结果决定下一步是重新规划还是继续执行
- 错误检测与恢复：Verifier能够发现执行失败，并触发重新规划以修复问题

运行前准备：
- 项目根目录创建 `.env` 并配置：
  - `LANGCHAIN_API_KEY`（用于 LangSmith 追踪，可选）
  - 如使用 ModelScope 接入：`MODELSCOPE_BASE_URL`、`MODELSCOPE_API_KEY`、`MODELSCOPE_MODEL_ID`

如何运行：
- 直接运行默认示例：`python 06_planner_executor_verifier.py`
- 更换任务请求：`python 06_planner_executor_verifier.py --request "查询苹果公司的研发支出和员工数量，计算人均研发支出"`
- 开启详细调试日志：`python 06_planner_executor_verifier.py --debug`

阅读建议：
- 先从"数据结构与模型定义"开始，理解PEV架构的状态管理
- 重点关注"核心节点实现"中的验证器节点，学习如何设计错误检测逻辑
- 最后看"工作流构建"，理解条件边如何实现动态路由
"""

import os
import json
import argparse
import re
from typing import List, TypedDict, Optional
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, END
from rich.console import Console
from rich.syntax import Syntax
from openai import OpenAI, RateLimitError, APIError
import logging
from rich.logging import RichHandler

# =========================
# 1) 数据结构与模型定义（Pydantic v2）
# =========================
class Plan(BaseModel):
    """计划模型：定义需要执行的步骤列表"""
    steps: List[str] = Field(description="需要执行的工具调用步骤列表")

class VerificationResult(BaseModel):
    """验证结果模型：定义验证器的输出"""
    is_failure: bool = Field(description="执行结果是否失败")
    reason: str = Field(description="失败原因或验证通过的说明")

class PlannerExecutorVerifierState(TypedDict):
    """规划→执行→验证架构的工作流状态：在节点间传递的共享数据"""
    user_request: str                      # 用户的原始请求
    plan: Optional[List[str]]             # Planner生成的执行计划
    intermediate_steps: List[str]         # 已执行的步骤结果
    verification_result: Optional[dict]   # Verifier的验证结果
    final_answer: Optional[str]           # 最终答案

# =========================
# 2) LLM 与控制台初始化
# =========================
console = Console()
DEBUG: bool = False  # 默认关闭调试模式
STREAM_TOKENS: bool = False  # 是否启用令牌流输出
MAX_STEPS: int = 10  # 最大执行步骤数

# 配置日志
logger = logging.getLogger("pev")
handler = RichHandler(console=console, rich_tracebacks=True, markup=True)
handler.setFormatter(logging.Formatter("%(message)s"))
logger.handlers = [handler]
logger.propagate = False
logger.setLevel(logging.DEBUG if DEBUG else logging.INFO)

class ModelScopeChat:
    """
    ModelScope的OpenAI兼容接口适配器：
    - 提供invoke(prompt)基本调用
    - 提供with_structured_output(PydanticModel)的结构化输出包装
    - 支持API错误时自动切换到备选模型
    """
    def __init__(self, base_url: str, api_key: str, model: str, fallback_model: Optional[str] = None, temperature: float = 0.2, extra_body: Optional[dict] = None):
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.model = model
        self.fallback_model = fallback_model
        self.base_url = base_url
        self.temperature = temperature
        self.extra_body = extra_body or {}
        self.switched = False  # 防止无限切换

    def invoke(self, prompt: str, stream_tokens: bool = False):
        try:
            if stream_tokens:
                resp_iter = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=self.temperature,
                    stream=True,
                    extra_body=self.extra_body,
                )
                buffer = []
                for chunk in resp_iter:
                    delta = getattr(chunk.choices[0], "delta", None)
                    token = getattr(delta, "content", "") if delta else ""
                    if token:
                        buffer.append(token)
                        console.print(token, end="")
                return "".join(buffer)
            else:
                resp = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=self.temperature,
                    stream=False,
                    extra_body=self.extra_body,
                )
                return resp.choices[0].message.content
        except (RateLimitError, APIError) as e:
            if not self.switched and self.fallback_model:
                console.print(f"[bold yellow]⚠️ 模型 {self.model} 请求失败: {str(e)}，尝试切换到备选模型 {self.fallback_model}[/bold yellow]")
                # 切换到备选模型
                self.model = self.fallback_model
                self.switched = True
                # 重新尝试请求
                return self.invoke(prompt, stream_tokens)
            else:
                # 如果没有备选模型或已经切换过，重新抛出异常
                raise

    def with_structured_output(self, pyd_model: type[BaseModel]):
        class _StructuredWrapper:
            def __init__(self, outer: "ModelScopeChat"):
                self.outer = outer

            def invoke(self, prompt: str) -> BaseModel:
                # 通过系统提示约束仅输出JSON
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
                
                # 验证并返回结果
                try:
                    parsed = pyd_model.model_validate(data)
                    if DEBUG:
                        console.print(f"[bold green]✅ 结构化解析成功[/bold green]：{pyd_model.__name__}")
                    return parsed
                except Exception as e:
                    console.print(f"[bold red]❌ 结构化解析失败: {e}[/bold red]")
                    raise

        return _StructuredWrapper(self)


def init_llm() -> ModelScopeChat:
    """
    初始化 ModelScope LLM（OpenAI 兼容接口）。
    - 可通过环境变量控制：
      MODELSCOPE_BASE_URL（默认：https://api-inference.modelscope.cn/v1）
      MODELSCOPE_API_KEY
      MODELSCOPE_MODEL_ID（默认：deepseek-ai/DeepSeek-V3.2）
      MODELSCOPE_MODEL_ID_R1（备选模型，可选）
    """
    base_url = os.environ.get("MODELSCOPE_BASE_URL", "https://api-inference.modelscope.cn/v1")
    api_key = os.environ.get("MODELSCOPE_API_KEY", "")
    model_id = os.environ.get("MODELSCOPE_MODEL_ID", "deepseek-ai/DeepSeek-V3.2")
    fallback_model = os.environ.get("MODELSCOPE_MODEL_ID_R1")
    
    # 为避免“未信任的 chat template”错误，增加信任参数；并请求 JSON 输出格式
    extra = {
        "enable_thinking": True,
        "trust_request_chat_template": True,
        "response_format": {"type": "json_object"},
    }
    
    return ModelScopeChat(
        base_url=base_url, 
        api_key=api_key, 
        model=model_id, 
        fallback_model=fallback_model,
        temperature=0.2, 
        extra_body=extra
    )

# =========================
# 2.5) 工具定义
# =========================
def flaky_web_search(query: str) -> str:
    """
    模拟一个不稳定的网络搜索工具，用于演示PEV架构的错误检测能力。
    - 当查询包含"employee count"时，模拟API失败
    - 其他查询则返回成功结果
    """
    console.print(f"--- TOOL: Searching for '{query}'... ---")
    if "employee count" or "员工数量" in query.lower():
        console.print("--- TOOL: [bold red]Simulating API failure![/bold red] ---")
        return "Error: Could not retrieve data. The API endpoint is currently unavailable."
    else:
        # 模拟搜索结果
        mock_results = {
            "apple r&d spend last fiscal year": "Apple's R&D spend in the last fiscal year (2023) was approximately $29 billion.",
            "apple annual revenue": "Apple's annual revenue in 2023 was approximately $383 billion.",
            "apple market share": "Apple's global smartphone market share in 2023 was approximately 18%.",
        }
        return mock_results.get(query.lower(), f"Search results for: {query}...")

# =========================
# 3) 核心节点实现
# =========================
def make_planner_node(llm: "ModelScopeChat"):
    """规划节点：根据用户请求生成执行计划"""
    planner_llm = llm.with_structured_output(Plan)

    def _node(state: PlannerExecutorVerifierState) -> dict:## 规划节点  
        console.print("--- [bold cyan]PLANNER: 生成任务计划[/bold cyan] ---")
        if DEBUG:
            console.print(f"[bold]输入请求：[/bold]{state['user_request']}")
        
        # 根据验证结果调整规划策略
        if state.get("verification_result") and state["verification_result"]["is_failure"]:
            # 如果验证失败，在规划时考虑失败原因
            prompt = f"""
            你是一个智能规划师。请根据用户请求和之前的失败原因，制定一个新的执行计划。
            
            用户请求：{state['user_request']}
            之前的失败原因：{state['verification_result']['reason']}
            
            请将请求分解为一系列需要使用'flaky_web_search'工具的查询步骤。
            每个步骤应该是一个明确的查询语句。
            尝试绕过之前失败的查询方式。
            """
        else:
            # 正常规划
            prompt = f"""
            你是一个智能规划师。请将用户请求分解为一系列需要使用'flaky_web_search'工具的查询步骤。
            
            用户请求：{state['user_request']}
            
            请将请求分解为一系列需要使用'flaky_web_search'工具的查询步骤。
            每个步骤应该是一个明确的查询语句。
            """
        
        plan = planner_llm.invoke(prompt)
        if DEBUG:
            console.print(f"[bold green]生成计划完成[/bold green]：{plan.steps}")
        
        return {"plan": plan.steps}

    return _node


def make_executor_node(llm: "ModelScopeChat"):
    """执行节点：执行计划中的下一步"""
    def _node(state: PlannerExecutorVerifierState) -> dict:
        console.print("--- [bold green]EXECUTOR: 执行下一步[/bold green] ---")
        if not state["plan"]:
            return {"intermediate_steps": state["intermediate_steps"]}
        
        next_step = state["plan"][0]
        if DEBUG:
            console.print(f"[bold]执行步骤：[/bold]{next_step}")
            
        result = flaky_web_search(next_step)
        
        # 更新状态：移除已执行的步骤，添加执行结果
        return {
            "plan": state["plan"][1:], 
            "intermediate_steps": state["intermediate_steps"] + [result]
        }

    return _node


def make_verifier_node(llm: "ModelScopeChat"):
    """验证节点：验证执行结果是否成功"""
    verifier_llm = llm.with_structured_output(VerificationResult)

    def _node(state: PlannerExecutorVerifierState) -> dict:
        console.print("--- [bold yellow]VERIFIER: 验证执行结果[/bold yellow] ---")
        
        # 获取最后一个执行结果
        last_result = state["intermediate_steps"][-1] if state["intermediate_steps"] else ""
        if DEBUG:
            console.print(f"[bold]最后执行结果：[/bold]{last_result}")
        
        prompt = f"""
        你是一个结果验证器。请检查以下执行结果是否成功完成了查询任务：
        
        执行结果：{last_result}
        
        如果结果包含"Error"或"API failure"等错误信息，则验证失败。
        否则，验证通过。
        """
        
        verification_result = verifier_llm.invoke(prompt)
        if DEBUG:
            console.print(f"[bold]验证结果：[/bold]{'失败' if verification_result.is_failure else '通过'}")
            console.print(f"[bold]失败原因：[/bold]{verification_result.reason}")
        
        if verification_result.is_failure:
            console.print("--- [bold yellow]VERIFIER: 检测到执行失败，触发重新规划[/bold yellow] ---")
        
        return {"verification_result": verification_result.model_dump()}

    return _node


def make_synthesizer_node(llm: "ModelScopeChat"):
    """汇总节点：根据所有执行结果生成最终答案"""
    def _node(state: PlannerExecutorVerifierState) -> dict:
        console.print("--- [bold magenta]SYNTHESIZER: 生成最终答案[/bold magenta] ---")
        
        context = "\n".join(state["intermediate_steps"])
        prompt = f"""
        请根据以下搜索结果，回答用户的问题：
        
        用户问题：{state['user_request']}
        搜索结果：
        {context}
        
        请提供一个清晰、简洁的最终答案。
        """
        
        if DEBUG:
            console.print(f"[bold]汇总上下文：[/bold]{context}")
        
        answer = llm.invoke(prompt, stream_tokens=STREAM_TOKENS)
        
        return {"final_answer": answer}

    return _node

# =========================
# 4) 构建与运行 LangGraph 工作流
# =========================
def build_app(llm: "ModelScopeChat"):
    """构建规划→执行→验证工作流：使用条件边实现动态路由"""
    graph_builder = StateGraph(PlannerExecutorVerifierState)
    
    # 添加节点
    graph_builder.add_node("planner", make_planner_node(llm))        # 规划节点
    graph_builder.add_node("executor", make_executor_node(llm))      # 执行节点
    graph_builder.add_node("verifier", make_verifier_node(llm))      # 验证节点
    graph_builder.add_node("synthesize", make_synthesizer_node(llm))  # 汇总节点
    
    # 设置入口点
    graph_builder.set_entry_point("planner")
    
    # 添加边
    graph_builder.add_edge("planner", "executor")  # 规划完成后执行第一步
    graph_builder.add_edge("executor", "verifier")  # 执行完成后验证结果
    
    # 添加条件边：根据验证结果决定下一步
    def route_after_verification(state: PlannerExecutorVerifierState) -> str:
        """
        路由函数：根据当前状态决定工作流的下一个节点
        
        参数：
            state: 当前工作流状态
            - state["plan"]: 剩余的执行计划（步骤列表）
            - state["verification_result"]: 验证结果（包含is_failure字段）
            
        返回值：
            下一个节点的名称："executor"、"planner" 或 "synthesize"
        """
        # 情况1：还有未执行的步骤且当前步骤验证通过
        if state["plan"] and not state["verification_result"]["is_failure"]:
            console.print(f"[bold yellow]路由决策：[/bold yellow]还有{len(state['plan'])}个步骤待执行，且当前步骤验证通过 → 继续执行下一个步骤")
            return "executor"
        # 情况2：当前步骤验证失败
        elif state["verification_result"]["is_failure"]:
            console.print("[bold yellow]路由决策：[/bold yellow]当前步骤验证失败 → 返回规划器重新规划")
            return "planner"
        # 情况3：所有步骤执行完成且验证通过
        else:
            console.print("[bold yellow]路由决策：[/bold yellow]所有步骤执行完成且验证通过 → 进入汇总阶段")
            return "synthesize"
    
    # 添加条件边：这是PEV架构的核心控制逻辑
    graph_builder.add_conditional_edges(
        "verifier",                  # 条件边的起始节点：验证器节点
        route_after_verification,    # 路由函数：根据状态决定下一个节点
        {                             # 路由映射：将路由函数返回值映射到实际节点名称
            "executor": "executor",  # 返回"executor" → 下一个节点是执行器
            "planner": "planner",    # 返回"planner" → 下一个节点是规划器
            "synthesize": "synthesize"  # 返回"synthesize" → 下一个节点是汇总器
        }
    )
    
    # 汇总完成后结束
    graph_builder.add_edge("synthesize", END)
    
    if DEBUG:
        console.print("[bold cyan]工作流构建完成[/bold cyan]")
        console.print("工作流路线：planner → executor → verifier → (条件分支)")
        console.print("条件分支：验证通过且有剩余步骤 → executor")
        console.print("条件分支：验证失败 → planner")
        console.print("条件分支：验证通过且无剩余步骤 → synthesize → END")
    
    return graph_builder.compile()


def run_workflow(app, user_request: str) -> PlannerExecutorVerifierState:
    """执行工作流并返回最终状态"""
    initial_input = {
        "user_request": user_request,
        "plan": None,
        "intermediate_steps": [],
        "verification_result": None,
        "final_answer": None
    }
    
    console.print(f"[bold cyan]🚀 启动规划→执行→验证工作流：[/bold cyan] '{user_request}'")
    
    final_state: Optional[PEVState] = None
    step = 0
    
    for state_update in app.stream(initial_input, stream_mode="values"):
        final_state = state_update
        step += 1
        
        if DEBUG:
            console.print(f"[bold]步骤 {step}[/bold] 当前状态：")
            console.print(f"  - 剩余计划：{state_update.get('plan', '无')}")
            console.print(f"  - 已执行步骤数：{len(state_update.get('intermediate_steps', []))}")
            if state_update.get('verification_result'):
                console.print(f"  - 验证结果：{'失败' if state_update['verification_result']['is_failure'] else '通过'}")
            if state_update.get('final_answer'):
                console.print(f"  - 最终答案：{state_update['final_answer'][:100]}...")
    
    console.print("[bold green]✅ PEV工作流完成[/bold green]")
    
    return final_state or initial_input

# =========================
# 5) 输出辅助：打印执行结果
# =========================
def print_execution_results(state: PlannerExecutorVerifierState) -> None:
    """打印执行结果的辅助函数"""
    console.print("--- ### 执行过程总结 ### ---")
    
    # 打印用户请求
    console.print(f"[bold]用户请求：[/bold]{state['user_request']}")
    
    # 打印执行步骤
    console.print("[bold]执行步骤：[/bold]")
    for i, result in enumerate(state['intermediate_steps']):
        console.print(f"  {i+1}. {result}")
    
    # 打印最终答案
    if state.get('final_answer'):
        console.print("\n[bold]最终答案：[/bold]")
        console.print(state['final_answer'])

# =========================
# 6) CLI 与入口
# =========================
def parse_args() -> argparse.Namespace:
    """命令行参数解析"""
    parser = argparse.ArgumentParser(
        description="规划→执行→验证架构：带错误检测与自修正的智能体工作流"
    )
    
    parser.add_argument(
        "--request",
        type=str,
        default="查询苹果公司上一财年的研发支出和员工数量，计算人均研发支出",
        help="用户请求（默认：查询苹果公司的研发支出和员工数量）",
    )
    
    parser.add_argument(
        "--debug",
        action="store_true",
        help="开启调试模式，输出详细日志",
    )
    
    parser.add_argument(
        "--stream",
        action="store_true",
        help="启用令牌流输出",
    )
    
    return parser.parse_args()


def main():
    """主函数"""
    global DEBUG, STREAM_TOKENS
    
    # 加载环境变量
    load_dotenv()
    
    # 解析命令行参数
    args = parse_args()
    DEBUG = args.debug
    STREAM_TOKENS = args.stream
    
    # 更新日志级别
    logger.setLevel(logging.DEBUG if DEBUG else logging.INFO)
    
    # 初始化LLM
    console.print("--- [bold blue]初始化LLM[/bold blue] ---")
    llm = init_llm()
    
    # 构建工作流
    console.print("--- [bold blue]构建工作流[/bold blue] ---")
    app = build_app(llm)
    
    # 执行工作流
    console.print("--- [bold blue]执行工作流[/bold blue] ---")
    final_state = run_workflow(app, args.request)
    
    # 打印结果
    console.print("--- [bold blue]输出结果[/bold blue] ---")
    print_execution_results(final_state)


if __name__ == "__main__":
    main()
# -*- coding: utf-8 -*-
"""
反思（Reflection）架构的可运行示例
 
学习目标（看完你能做到什么）：
- 读懂什么是“反思式”智能体：先生成，再评审，最后改写，使答案更可靠
- 理解 LangGraph 如何把多步逻辑编排成“有状态的工作流”
- 学会用 Pydantic v2 约束 LLM 输出为结构化数据（更稳、更好用）
- 能把脚本作为命令行程序运行，并按需保存改写后的代码
 
核心概念速览：
- StateGraph：有状态的“流程图”，每个节点是一个函数，节点之间按边连接顺序执行
- 结构化输出（with_structured_output）：让 LLM 根据我们定义的“数据模型”吐出 JSON，再自动解析成 Python 对象
- 三步节点：生成（Generator）→ 评审（Critic）→ 改写（Refiner）
 
运行前准备：
- 项目根目录创建 `.env` 并配置：
  - `LANGCHAIN_API_KEY`（用于 LangSmith 追踪，可选）
  - 如使用 ModelScope 接入：`MODELSCOPE_BASE_URL`、`MODELSCOPE_API_KEY`、`MODELSCOPE_MODEL_ID`
 
如何运行：
- 直接运行默认示例：`python 01_reflection.py`
- 更换任务请求：`python 01_reflection.py --request "Write a Python function to sort a list."`
- 保存改写后的代码：`python 01_reflection.py --save-refined refined.py`
 
阅读建议：
- 先从“数据模型”和“三个节点函数”开始理解，再看“build_app”和“run_workflow”如何把所有部件串起来
"""

import os
import json                                                                           
import argparse
from typing import List, TypedDict, Optional

from dotenv import load_dotenv

from pydantic import BaseModel, Field

# LangGraph
from langgraph.graph import StateGraph, END

# 控制台美化
from rich.console import Console
from rich.syntax import Syntax
 
from openai import OpenAI
import logging
from rich.logging import RichHandler

# =========================
# 1) 数据结构与模型定义（Pydantic v2）
# =========================
class DraftCode(BaseModel):
    """初稿：生成的代码与简要说明"""
    code: str = Field(description="为用户请求生成的 Python 代码")
    explanation: str = Field(description="代码工作原理的简要说明")


class Critique(BaseModel):
    """评审：结构化的代码批判与改进建议"""
    has_errors: bool = Field(description="是否存在潜在错误或逻辑问题")
    is_efficient: bool = Field(description="实现是否高效、是否符合最佳实践")
    suggested_improvements: List[str] = Field(description="可执行的、具体的改进建议")
    critique_summary: str = Field(description="评审摘要")


class RefinedCode(BaseModel):
    """改写：基于评审意见的最终版本代码"""
    refined_code: str = Field(description="改进后的最终 Python 代码")
    refinement_summary: str = Field(description="根据评审所做改动的说明")


class ReflectionState(TypedDict):
    """工作流状态：在节点间传递的共享数据"""
    # 用户的原始请求文本，用于驱动“生成初稿”节点
    user_request: str
    # “生成初稿”节点输出的结构化结果，通常包含 code 与 explanation
    draft: Optional[dict]
    # “评审”节点输出的结构化评审信息（是否有错误、效率、建议、摘要）
    critique: Optional[dict]
    # “改写”节点输出的最终代码与改动摘要（refined_code 与 refinement_summary）
    refined_code: Optional[dict]


# =========================
# 2) LLM 与控制台初始化
# =========================
console = Console()
DEBUG: bool = True  # 通过 --debug 开启更详细的日志
logger = logging.getLogger("reflection")
handler = RichHandler(console=console, rich_tracebacks=True, markup=True)
handler.setFormatter(logging.Formatter("%(message)s"))
logger.handlers = [handler]
logger.propagate = False
logger.setLevel(logging.DEBUG if DEBUG else logging.INFO)

class ModelScopeChat:
    """
    ModelScope 的 OpenAI 兼容接口适配器：
    - 提供 invoke(prompt) 基本调用
    - 提供 with_structured_output(PydanticModel) 的结构化输出包装
 
初学者理解要点：
- 为什么需要“适配器”？因为我们的工作流依赖“结构化输出”，而不少服务默认只返回纯文本。
- 适配器会尽量要求模型“只输出 JSON”，再解析为 Pydantic v2 模型；这样后续节点就能稳稳地拿到字段，而不是杂乱的文本。
    """
    def __init__(self, base_url: str, api_key: str, model: str, temperature: float = 0.2, extra_body: Optional[dict] = None):
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.model = model
        self.base_url = base_url
        self.temperature = temperature
        self.extra_body = extra_body or {}

    def invoke(self, prompt: str):
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
            stream=False,
            extra_body=self.extra_body,
        )
        # 非流式返回：choices[0].message.content
        return resp.choices[0].message.content

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
                resp = self.outer.client.chat.completions.create(
                    model=self.outer.model,
                    messages=messages,
                    temperature=self.outer.temperature,
                    stream=False,
                    extra_body=self.outer.extra_body,
                )
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
                    # DraftCode: 常见返回 'function' 字段，映射到 'code'
                    if "code" not in data and "function" in data:
                        data["code"] = data.pop("function")
                        mappings_applied.append("function → code")
                    if "explanation" not in data and "desc" in data:
                        data["explanation"] = data.pop("desc")
                        mappings_applied.append("desc → explanation")
                    # RefinedCode: 将 'code' 映射为 'refined_code'
                    if "refined_code" not in data and "code" in data:
                        data["refined_code"] = data.pop("code")
                        mappings_applied.append("code → refined_code")
                    # Critique: 可能给 'summary' 映射为 'critique_summary'
                    if "critique_summary" not in data and "summary" in data:
                        data["critique_summary"] = data.pop("summary")
                        mappings_applied.append("summary → critique_summary")
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
 
小贴士：
- 直接在浏览器访问 Base URL 返回 404 正常，客户端会在此基础上拼接具体路径（如 /chat/completions）。
- 请把真实的 Token 写在 `.env` 中，不要硬编码到脚本。
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
    return ModelScopeChat(base_url=base_url, api_key=api_key, model=model_id, temperature=0.2, extra_body=extra)


# =========================
# 3) 三个核心节点：生成→评审→改写
# =========================
def make_generator_node(llm: "ModelScopeChat"):
    """生成节点：根据用户请求产出第一版代码（结构化输出为 DraftCode）"""
    generator_llm = llm.with_structured_output(DraftCode)

    def _node(state: ReflectionState) -> dict:
        console.print("--- 1. 生成初稿 ---")
        if DEBUG:
            console.print(f"[bold]输入请求：[/bold]{state['user_request']}")
        prompt = f"""你是资深 Python 程序员。请为下面的请求编写函数，并给出简要说明：

请求：{state['user_request']}
"""
        draft = generator_llm.invoke(prompt)
        if DEBUG:
            console.print("[bold green]生成初稿完成（结构化）[/bold green]")
        return {"draft": draft.model_dump()}

    return _node


def make_critic_node(llm: "ModelScopeChat"):
    """评审节点：针对初稿进行结构化评审（结构化输出为 Critique）"""
    critic_llm = llm.with_structured_output(Critique)

    def _node(state: ReflectionState) -> dict:
        console.print("--- 2. 评审初稿 ---")
        if DEBUG:
            console.print("[bold]待评审代码片段（前120字符）：[/bold]")
            console.print((state["draft"]["code"] or "")[:120] + "...")
        code_to_critique = state["draft"]["code"]
        prompt = f"""你是一名资深 Python 代码审阅者。请对下方代码进行严格评审，并输出结构化建议：
1) 是否存在潜在错误或未覆盖的边界？
2) 是否遵循最佳实践并具备高效率？
3) 给出可执行的改进建议清单。

代码：
```python
{code_to_critique}
```
"""
        critique = critic_llm.invoke(prompt)
        if DEBUG:
            console.print("[bold green]评审完成（结构化）[/bold green]")
        return {"critique": critique.model_dump()}

    return _node


def make_refiner_node(llm: "ModelScopeChat"):
    """改写节点：结合评审建议重写代码（结构化输出为 RefinedCode）"""
    refiner_llm = llm.with_structured_output(RefinedCode)

    def _node(state: ReflectionState) -> dict:
        console.print("--- 3. 改写代码 ---")
        if DEBUG:
            console.print("[bold]将根据以下评审建议进行改写（字段视图）：[/bold]")
            console.print(list(state["critique"].keys()))
        draft_code = state["draft"]["code"]
        critique_suggestions = json.dumps(state["critique"], ensure_ascii=False, indent=2)
        prompt = f"""你是一名资深 Python 程序员。请基于评审意见重写代码：

原始代码：
```python
{draft_code}
```

评审建议（结构化）：
{critique_suggestions}

请输出最终改进后的代码与改动摘要。
"""
        refined = refiner_llm.invoke(prompt)
        if DEBUG:
            console.print("[bold green]改写完成（结构化）[/bold green]")
        return {"refined_code": refined.model_dump()}

    return _node


# =========================
# 4) 构建与运行 LangGraph 工作流
# =========================
def build_app(llm: "ModelScopeChat"):
    """构建线性工作流：entry → generator → critic → refiner → END"""
    graph_builder = StateGraph(ReflectionState)  # 创建“流程图”构建器，并声明状态数据的类型

    graph_builder.add_node("generator", make_generator_node(llm))  # 第 1 步：生成初稿（Draft）
    graph_builder.add_node("critic", make_critic_node(llm))        # 第 2 步：评审初稿（Critique）
    graph_builder.add_node("refiner", make_refiner_node(llm))      # 第 3 步：改写代码（Refined）

    graph_builder.set_entry_point("generator")      # 指定入口节点为“生成初稿”
    graph_builder.add_edge("generator", "critic")   # 流程：生成 → 评审
    graph_builder.add_edge("critic", "refiner")     # 流程：评审 → 改写
    graph_builder.add_edge("refiner", END)          # 流程：改写 → 结束（返回最终结果）
    if DEBUG:
        console.print("[bold cyan]工作流编排：[/bold cyan] generator → critic → refiner → END")

    return graph_builder.compile()  # 编译成可执行的“应用”，供 run_workflow 调用


def run_workflow(app, user_request: str) -> ReflectionState:
    """执行工作流并返回最终状态"""
    initial_input = {"user_request": user_request}
    console.print(f"[bold cyan]🚀 启动反思工作流：[/bold cyan] '{user_request}'")
    final_state: Optional[ReflectionState] = None
    step = 0
    for state_update in app.stream(initial_input, stream_mode="values"):
        final_state = state_update  # 流式累积到最终状态
        step += 1
        if DEBUG:
            console.print(f"[bold]步骤 {step}[/bold] 当前状态字段：{list(state_update.keys())}")
    console.print("[bold green]✅ 工作流完成[/bold green]")
    return final_state or initial_input  # 兜底


# =========================
# 5) 输出辅助：打印前后对比
# =========================
def print_before_after(state: ReflectionState) -> None:
    """打印初稿/评审/改写后的代码，便于直观对比"""
    console.print("--- ### 初稿（Draft） ---")
    explanation = state.get("draft", {}).get("explanation", "")
    if explanation:
        console.print(f"[bold]说明：[/bold]{explanation}")
    code = state.get("draft", {}).get("code", "")
    if code:
        console.print(Syntax(code, "python", theme="monokai", line_numbers=True))

    console.print("--- ### 评审（Critique） ---")
    critique = state.get("critique", {})
    if critique:
        console.print(json.dumps(critique, ensure_ascii=False, indent=2))

    console.print("--- ### 改写后（Refined） ---")
    refined = state.get("refined_code", {}).get("refined_code", "")
    if refined:
        console.print(Syntax(refined, "python", theme="monokai", line_numbers=True))


# =========================
# 6) CLI 与入口
# =========================
def parse_args() -> argparse.Namespace:
    """命令行参数解析"""
    parser = argparse.ArgumentParser(
        description="反思（Reflection）架构：生成→评审→改写的可运行脚本"
    )
    parser.add_argument(
        "--request",
        type=str,
        default="Write a Python function to find the nth Fibonacci number.",
        help="用户请求（默认：斐波那契函数）",
    )
    parser.add_argument(
        "--save-refined",
        type=str,
        default="",
        help="将改写后的代码保存到指定文件路径（可选）",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="开启详细教学日志（结构化提示、状态变化、片段预览）",
    )
    return parser.parse_args()


def main():
    load_dotenv()
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_PROJECT"] = "Agentic Architecture - Reflection (ModelScope)"

    if not os.environ.get("MODELSCOPE_API_KEY"):
        console.print("[bold red]MODELSCOPE_API_KEY 未设置，请在项目根目录配置 .env[/bold red]")
    if not os.environ.get("LANGCHAIN_API_KEY"):
        console.print("[bold yellow]提示：未设置 LANGCHAIN_API_KEY，LangSmith 追踪将不可用[/bold yellow]")

    args = parse_args()
    global DEBUG
    DEBUG = bool(args.debug)
    llm = init_llm()
    if DEBUG:
        console.print("[bold cyan]推理服务配置[/bold cyan]:")
        console.print(f"base_url={llm.base_url}")
        console.print(f"model_id={llm.model}")
    app = build_app(llm)

    final_state = run_workflow(app, args.request)
    print_before_after(final_state)

    # 可选：保存改写后的代码到文件
    if args.save_refined:
        refined_code = final_state.get("refined_code", {}).get("refined_code", "")
        if refined_code:
            with open(args.save_refined, "w", encoding="utf-8") as f:
                f.write(refined_code)
            console.print(f"[bold green]已保存改写代码至：[/bold green]{args.save_refined}")
        else:
            console.print("[bold red]未找到改写后的代码，保存失败[/bold red]")


if __name__ == "__main__":
    main()


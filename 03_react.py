# -*- coding: utf-8 -*-
"""
ReAct（Reason + Act）架构的可运行示例
 
学习目标（看完你能做到什么）：
- 了解“先思考，再行动，再观察”的 ReAct 交互模式
- 使用 LangGraph 编排“循环式工作流”，在达到终止条件前持续迭代
- 用 Pydantic v2 让 LLM 以结构化 JSON 返回“想法/行动/是否结束”
- 在纯本地工具下学习 ReAct 的实现思路，保证可运行、易理解
 
核心概念速览：
- ReAct 循环：Reasoner（思考）→ Actor（行动）→ 观察（Observation）→ 下一轮 Reasoner
- 结构化输出：约束 LLM 仅输出我们定义的数据结构，提高解析稳定性
- 条件边与终止：当 `is_final=True` 或达到迭代上限时结束循环
 
运行前准备：
- 项目根目录创建 `.env` 并配置：
  - `MODELSCOPE_API_KEY`（必需，ModelScope 推理令牌）
  - 可选：`MODELSCOPE_BASE_URL`（默认 `https://api-inference.modelscope.cn/v1`）、`MODELSCOPE_MODEL_ID`（默认 `deepseek-ai/DeepSeek-V3.2`）
  - 可选：`LANGCHAIN_API_KEY`（用于 LangSmith 追踪）
 
如何运行：
- 默认示例：`python 03_react.py`
- 指定问题：`python 03_react.py --question "请计算表达式 12*(3+4) 并给出结果的文字说明"`
- 开启令牌流与教学日志：`python 03_react.py --stream --debug`
 
阅读建议：
- 先看“数据模型与状态”与“ModelScope 适配器”，再看“Reasoner/Actor 节点”与“工作流编排与循环”
- 工具均为本地实现（安全），有利于专注学习 ReAct 的交互逻辑
"""
import os
import json
import argparse
from typing import List, TypedDict, Optional, Dict, Any
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, END
from rich.console import Console
from rich.syntax import Syntax
from openai import OpenAI
import logging
from rich.logging import RichHandler

console = Console()
DEBUG: bool = True
STREAM_TOKENS: bool = False
MAX_ITERATIONS: int = 6

logger = logging.getLogger("react")
handler = RichHandler(console=console, rich_tracebacks=True, markup=True)
handler.setFormatter(logging.Formatter("%(message)s"))
logger.handlers = [handler]
logger.propagate = False
logger.setLevel(logging.DEBUG if DEBUG else logging.INFO)

class ModelScopeChat:
    """
    ModelScope 的 OpenAI 兼容接口适配器：
    - 提供 invoke(prompt, stream_tokens) 基本调用
    - 提供 with_structured_output(PydanticModel) 的结构化输出包装
 
初学者理解要点：
- 我们通过系统消息明确字段与类型，让模型“只输出 JSON 对象”，尽量避免自由文本
- 返回后使用 Pydantic v2 验证与字段别名兜底，提高健壮性
    """
    def __init__(self, base_url: str, api_key: str, model: str, temperature: float = 0.2, extra_body: Optional[dict] = None):
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.model = model
        self.base_url = base_url
        self.temperature = temperature
        self.extra_body = extra_body or {}
    def invoke(self, prompt: str, stream_tokens: bool = False) -> str:
        # 非结构化调用：可选择令牌流（实时打印），适合调试思考过程
        if stream_tokens:
            resp_iter = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                stream=True,
                extra_body=self.extra_body,
            )
            parts = []
            import sys as _sys
            for ch in resp_iter:
                delta = getattr(ch.choices[0], "delta", None)
                token = getattr(delta, "content", "") if delta else ""
                if token:
                    parts.append(token)
                    _sys.stdout.write(token)
                    _sys.stdout.flush()
            return "".join(parts)
        else:
            # 非流式：一次性返回文本
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                stream=False,
                extra_body=self.extra_body,
            )
            return resp.choices[0].message.content or ""
    def with_structured_output(self, pyd_model: type[BaseModel]):
        class _StructuredWrapper:
            def __init__(self, outer: "ModelScopeChat"):
                self.outer = outer
            def invoke(self, prompt: str) -> BaseModel:
                # 生成“字段/类型约束”的系统消息，提升 JSON 结构化输出稳定性
                schema = pyd_model.model_json_schema()
                props = schema.get("properties", {})
                required = schema.get("required", [])
                schema_text_lines = []
                for k, v in props.items():
                    t = v.get("type", "string")
                    schema_text_lines.append(f"- {k}: {t}")
                schema_text = "\n".join(schema_text_lines) or "- fields"
                required_text = ", ".join(required) if required else "all"
                system_msg = (
                    "只输出一个 JSON 对象，严格匹配以下字段与类型：\n"
                    f"{schema_text}\n"
                    f"必须包含字段：{required_text}\n"
                    "不要输出任何解释或多余文本。"
                )
                messages = [
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": prompt},
                ]
                if STREAM_TOKENS:
                    # 结构化令牌流：边接收边打印，便于观察模型如何填充 JSON
                    content_iter = self.outer.client.chat.completions.create(
                        model=self.outer.model,
                        messages=messages,
                        temperature=self.outer.temperature,
                        stream=True,
                        extra_body=self.outer.extra_body,
                    )
                    import sys as _sys
                    _sys.stdout.write("\n📡 正在接收结构化 JSON...\n")
                    _sys.stdout.flush()
                    parts = []
                    for chunk in content_iter:
                        delta = getattr(chunk.choices[0], "delta", None)
                        token = getattr(delta, "content", "") if delta else ""
                        if token:
                            parts.append(token)
                            _sys.stdout.write(token)
                            _sys.stdout.flush()
                    content = "".join(parts)
                else:
                    # 一次性返回 JSON 字符串
                    resp = self.outer.client.chat.completions.create(
                        model=self.outer.model,
                        messages=messages,
                        temperature=self.outer.temperature,
                        stream=False,
                        extra_body=self.outer.extra_body,
                    )
                    content = resp.choices[0].message.content or ""
                import json as _json, re
                from pydantic import ValidationError
                def _extract_json(s: str) -> str:
                    # 宽松提取：从文本中截取最外层 {...} 或 [...]
                    m = re.search(r'(\{[\s\S]*\}|\[[\s\S]*\])', s)
                    return m.group(1) if m else "{}"
                raw = content.strip()
                try:
                    # 首选：直接解析完整 JSON
                    data = _json.loads(raw)
                except Exception:
                    # 退路：使用宽松提取后的片段再解析
                    data = _json.loads(_extract_json(raw))
                try:
                    return pyd_model.model_validate(data)
                except ValidationError:
                    # 字段兜底映射
                    # 当模型使用了常见同义字段名时，进行稳定的别名回填，确保 Pydantic 校验通过
                    # 将常见别名“reason→thought”、“tool_name→action”、“arguments→action_input”、“answer→final_answer”
                    if "thought" not in data and "reason" in data:
                        data["thought"] = data.pop("reason")
                    if "action" not in data and "tool_name" in data:
                        data["action"] = data.pop("tool_name")
                    if "action_input" not in data and "arguments" in data:
                        data["action_input"] = data.pop("arguments")
                    if "final_answer" not in data and "answer" in data:
                        data["final_answer"] = data.pop("answer")
                    ai = data.get("action_input", None)
                    act = data.get("action", None)
                    if not isinstance(ai, dict):
                        if isinstance(ai, str):
                            try:
                                parsed = _json.loads(ai)
                                if isinstance(parsed, dict):
                                    data["action_input"] = parsed
                                else:
                                    raise Exception()
                            except Exception:
                                if act == "calc":
                                    data["action_input"] = {"expression": ai}
                                elif act == "normalize_text":
                                    data["action_input"] = {"text": ai}
                                elif act == "current_time":
                                    data["action_input"] = {"format": ai}
                                else:
                                    data["action_input"] = {"input": ai}
                        else:
                            data["action_input"] = {"input": ai} if ai is not None else {}
                    return pyd_model.model_validate(data)
        return _StructuredWrapper(self)

def init_llm() -> ModelScopeChat:
    """
    初始化 ModelScope LLM（OpenAI 兼容接口）。
    - MODELSCOPE_BASE_URL（默认：https://api-inference.modelscope.cn/v1）
    - MODELSCOPE_API_KEY
    - MODELSCOPE_MODEL_ID（默认：deepseek-ai/DeepSeek-V3.2）
    - 强制 JSON 输出以配合结构化解析
    """
    base_url = os.environ.get("MODELSCOPE_BASE_URL", "https://api-inference.modelscope.cn/v1")
    api_key = os.environ.get("MODELSCOPE_API_KEY", "")
    model_id = os.environ.get("MODELSCOPE_MODEL_ID", "deepseek-ai/DeepSeek-V3.2")
    extra = {
        "enable_thinking": True,
        "trust_request_chat_template": True,
        "response_format": {"type": "json_object"},
    }
    return ModelScopeChat(base_url=base_url, api_key=api_key, model=model_id, temperature=0.2, extra_body=extra)

# =========================
# 工具集合（本地、安全）
# =========================
def _safe_eval(expr: str) -> float:
    """
    安全算术表达式求值：仅支持 + - * / ** 和一元负号
    避免任意代码执行风险
    """
    import ast, operator as op
    allowed = {
        ast.Add: op.add,
        ast.Sub: op.sub,
        ast.Mult: op.mul,
        ast.Div: op.truediv,
        ast.Pow: op.pow,
        ast.USub: op.neg,
    }
    def _eval(node):
        # 递归计算：仅允许定义好的 AST 节点与运算符
        if isinstance(node, ast.Num):
            return node.n
        if isinstance(node, ast.BinOp) and type(node.op) in allowed:
            return allowed[type(node.op)](_eval(node.left), _eval(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in allowed:
            return allowed[type(node.op)](_eval(node.operand))
        raise ValueError("unsupported expression")
    tree = ast.parse(expr, mode="eval")
    return float(_eval(tree.body))

def tool_calc(arguments: Dict[str, Any]) -> Dict[str, Any]:
    """工具：计算算术表达式的值"""
    expr = str(arguments.get("expression", ""))
    try:
        # 使用受控的 _safe_eval 执行表达式计算
        value = _safe_eval(expr)
        return {"expression": expr, "value": value}
    except Exception as e:
        return {"expression": expr, "error": str(e)}

def tool_normalize_text(arguments: Dict[str, Any]) -> Dict[str, Any]:
    """工具：标准化文本（小写、去标点、压缩空白）"""
    import re
    text = str(arguments.get("text", ""))
    # 小写化 → 去标点（替换为空格）→ 压缩连续空白
    lower = text.lower()
    no_punct = re.sub(r"[^\w\s]", " ", lower)
    normalized = re.sub(r"\s+", " ", no_punct).strip()
    return {"normalized": normalized, "orig": text}

def tool_current_time(arguments: Dict[str, Any]) -> Dict[str, Any]:
    """工具：返回当前本地时间字符串"""
    from datetime import datetime
    fmt = str(arguments.get("format", "%Y-%m-%d %H:%M:%S"))
    return {"now": datetime.now().strftime(fmt), "format": fmt}

TOOLS_REGISTRY = {
    "calc": tool_calc,
    "normalize_text": tool_normalize_text,
    "current_time": tool_current_time,
}

# =========================
# 数据模型与状态
# =========================
class ReActStep(BaseModel):
    """结构化的 ReAct 步骤输出：想法/行动/输入/是否结束/最终答案"""
    thought: str = Field(description="思考过程的简要说明")
    action: Optional[str] = Field(default=None, description="选择的工具名（可选）")
    action_input: Optional[Dict[str, Any]] = Field(default=None, description="传给工具的参数对象（可选）")
    is_final: bool = Field(description="是否为最终回答（True 则不再行动）")
    final_answer: Optional[str] = Field(default=None, description="最终回答（当 is_final=True 时应提供）")

class ReActState(TypedDict):
    """工作流状态：在节点间传递的共享数据"""
    question: str
    steps: List[dict]              # 每轮的结构化步骤与观察
    final_answer: Optional[str]    # 最终回答（若已结束）

# =========================
# Reasoner 与 Actor 节点
# =========================
def make_reasoner_node(llm: "ModelScopeChat"):
    """
    Reasoner：给模型上下文（问题 + 过往观察），让其“先思考”，
    再决定是否选择一个工具执行；或者直接输出最终答案。
    """
    reasoner_llm = llm.with_structured_output(ReActStep)
    def _node(state: ReActState) -> dict:
        question = state["question"]
        history = state.get("steps", [])
        # 提供“可用工具目录”，引导模型选择合法工具与参数（名称需与 TOOLS_REGISTRY 对齐）
        tool_catalog = json.dumps(
            {
                "available_tools": [
                    {"name": "calc", "args": {"expression": "字符串算术表达式，如 '12*(3+4)'" }},
                    {"name": "normalize_text", "args": {"text": "要标准化的文本"}},
                    {"name": "current_time", "args": {"format": "可选时间格式"}},
                ]
            },
            ensure_ascii=False,
            indent=2,
        )
        # 将历史步骤（含上一轮的观察）传入，供“思考”参考
        history_text = json.dumps(history, ensure_ascii=False, indent=2)
        # 明确要求输出严格 JSON，字段与类型固定，减少解析歧义
        prompt = (
            "你是一个 ReAct 智能体：先思考（thought），再选择行动（action/action_input），或直接给出最终答案。\n"
            "请严格输出 JSON，字段：thought, action(可选), action_input(可选), is_final(必填), final_answer(可选)。\n"
            "若选择行动，只能从给定工具中选一个；若直接给出最终答案，设置 is_final=true 并填写 final_answer。\n\n"
            f"工具目录：\n{tool_catalog}\n\n"
            f"问题：\n{question}\n\n"
            f"过往步骤与观察（供参考）：\n{history_text}\n"
        )
        # 返回结构化的“思考/决策”，并把该步加入状态序列
        step = reasoner_llm.invoke(prompt)
        return {"steps": state["steps"] + [step.model_dump()]}
    return _node

def make_actor_node():
    """
    Actor：如有行动（action），执行对应的本地工具，并记录 observation；
    如 is_final=True，则不执行任何工具，仅保留模型最终回答。
    """
    def _node(state: ReActState) -> dict:
        steps = list(state.get("steps", []))
        last = steps[-1] if steps else {}
        action = last.get("action")
        action_input = last.get("action_input") or {}
        is_final = bool(last.get("is_final"))
        final_answer = last.get("final_answer") or ""
        if is_final:
            # 直接输出最终答案，不再行动
            return {"final_answer": final_answer}
        if not action:
            # 没有指定工具，则记录空观察
            last["observation"] = {"info": "no_action"}
            steps[-1] = last
            return {"steps": steps}
        func = TOOLS_REGISTRY.get(action)
        if func is None:
            # 非法工具名：记录错误观察，便于下一轮思考调整
            last["observation"] = {"error": f"unknown tool '{action}'"}
            steps[-1] = last
            return {"steps": steps}
        try:
            # 执行本地工具，产出结构化观察
            output = func(action_input)
            last["observation"] = output
        except Exception as e:
            last["observation"] = {"error": str(e)}
        steps[-1] = last
        return {"steps": steps}
    return _node

# =========================
# 条件判断：继续循环还是结束
# =========================
def should_continue(state: ReActState) -> str:
    """
    条件边映射函数：
    - 若最后一步 is_final=True → 结束
    - 若迭代次数达到上限 → 结束
    - 否则继续 Reasoner
    """
    steps = state.get("steps", [])
    if steps:
        last = steps[-1]
        if bool(last.get("is_final")):
            return "end"
    if len(steps) >= MAX_ITERATIONS:
        # 防止无限循环：达到上限后强制结束
        return "end"
    return "reason"

# =========================
# 构建与运行工作流
# =========================
def build_app(llm: "ModelScopeChat"):
    """
    构建循环工作流：reason → act → [继续/结束]
    - 使用条件边在 act 后决定下个节点（继续 reason 或 END）
    设计意图：
    - “思考”节点只负责结构化决策（thought / action / is_final / final_answer）
    - “行动”节点只负责基于决策执行本地工具并写入 observation
    - 条件边是关键：根据最新步骤是否 is_final 或是否达到上限来决定是否继续
    伪代码（简化表达）：
      state = {question, steps: [], final_answer: None}
      loop:
        step = reason(state)              # 产出结构化 {thought, action, action_input, is_final, final_answer}
        state = act(state + step)         # 执行工具并写入 observation；或直接写入 final_answer
        if is_final(step) or reach_limit: # 终止条件
            break
        else:
            continue                      # 回到 reason
    """
    graph = StateGraph(ReActState)
    # 添加两个核心节点：思考（reason）与行动（act）
    graph.add_node("reason", make_reasoner_node(llm))
    graph.add_node("act", make_actor_node())
    # 设置入口为“思考”，然后固定边 reason→act
    graph.set_entry_point("reason")
    graph.add_edge("reason", "act")
    # 条件边：act 之后根据状态决定继续 reason 或 END
    def _decide_next(state: ReActState):
        # 根据 should_continue 的返回值（'reason' 或 'end'）进行路由
        # 注意：add_conditional_edges 的第三个参数提供了标签到节点/END 的映射
        # - 'reason' → 节点 'reason'（继续循环）
        # - 'end'    → END（终止）
        return should_continue(state)
    graph.add_conditional_edges("act", _decide_next, {"reason": "reason", "end": END})
    return graph.compile()

def run_workflow(app, question: str) -> ReActState:
    """
    执行工作流并返回最终状态；打印状态字段变化便于学习理解
    - 初始状态包含：question 与空 steps
    - 流式执行：reason → act → reason → ... → END
    关键点：
    - 使用 app.stream 获取每一步的状态增量，可直观看到状态键的变化
    - DEBUG 模式下，打印最近一步的结构化内容（含 thought / action / observation）
    - 工作流结束后打印“一次会话回放”和“最终回答”，便于教学展示
    伪代码（状态打印逻辑）：
      initial = {question, steps: [], final_answer: None}
      for update in app.stream(initial):
          print(keys(update))              # 可见的键
          last = tail(update.steps)        # 最近一步
          print(JSON(last))                # 思考/行动/观察预览
      print("完成")
    """
    initial = {"question": question, "steps": [], "final_answer": None}
    final_state: Optional[ReActState] = None
    # 以流式方式逐步获取状态增量，每次包含当前可见的键值
    for update in app.stream(initial, stream_mode="values"):
        final_state = update
        if DEBUG:
            console.print(f"[bold]状态更新：[/bold]{list(update.keys())}")
            # 展示最近一步（thought/action/observation）
            last = (update.get("steps") or [])[-1] if (update.get("steps") or []) else {}
            if last:
                preview = json.dumps(last, ensure_ascii=False, indent=2)
                console.print(preview)
    console.print("[bold green]✅ ReAct 工作流完成[/bold green]")
    return final_state or initial

# =========================
# 输出辅助
# =========================
def print_session_summary(state: ReActState) -> None:
    """打印所有步骤与最终回答，便于直观教学展示"""
    console.print("--- ### ReAct 步骤回放 ---")
    steps = state.get("steps", [])
    if steps:
        console.print(json.dumps(steps, ensure_ascii=False, indent=2))
    console.print("--- ### 最终回答 ---")
    ans = state.get("final_answer") or ""
    if ans:
        console.print(ans)

# =========================
# CLI 与入口
# =========================
def parse_args() -> argparse.Namespace:
    """命令行参数解析：问题、教学日志、令牌流、迭代上限"""
    parser = argparse.ArgumentParser(description="ReAct（Reason + Act）架构：可运行学习脚本")
    parser.add_argument(
        "--question",
        type=str,
        default="请计算表达式 12*(3+4)，并用一句话说明结果。",
        help="用户问题（ReAct 将按需选择工具或直接给出答案）",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="开启详细教学日志",
    )
    parser.add_argument(
        "--stream",
        action="store_true",
        help="实时打印模型令牌（结构化 JSON 也将边接收边展示）",
    )
    parser.add_argument(
        "--max-iters",
        type=int,
        default=6,
        help="ReAct 循环的最大迭代次数（达到上限将强制结束）",
    )
    return parser.parse_args()

def main():
    load_dotenv()
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_PROJECT"] = "Agentic Architecture - ReAct (ModelScope)"
    if not os.environ.get("MODELSCOPE_API_KEY"):
        console.print("[bold red]MODELSCOPE_API_KEY 未设置，请在项目根目录配置 .env[/bold red]")
    if not os.environ.get("LANGCHAIN_API_KEY"):
        console.print("[bold yellow]提示：未设置 LANGCHAIN_API_KEY，LangSmith 追踪将不可用[/bold yellow]")
    args = parse_args()
    global DEBUG, STREAM_TOKENS, MAX_ITERATIONS
    DEBUG = bool(args.debug)
    STREAM_TOKENS = bool(args.stream)
    try:
        MAX_ITERATIONS = int(args.max_iters)
    except Exception:
        MAX_ITERATIONS = 6
    llm = init_llm()
    if DEBUG:
        console.print("[bold cyan]推理服务配置[/bold cyan]:")
        console.print(f"base_url={llm.base_url}")
        console.print(f"model_id={llm.model}")
    app = build_app(llm)
    final_state = run_workflow(app, args.question)
    # Reasoner 可能在最后一步直接写入 final_answer；若空则尝试从最后一步读出
    if not final_state.get("final_answer"):
        steps = final_state.get("steps") or []
        if steps and steps[-1].get("final_answer"):
            final_state["final_answer"] = steps[-1]["final_answer"]
    print_session_summary(final_state)

if __name__ == "__main__":
    main()

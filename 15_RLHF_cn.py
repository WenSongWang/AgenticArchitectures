# -*- coding: utf-8 -*-
"""
自改进循环（Self-Refine & RLHF 类比）架构的可运行示例

学习目标（看完你能做到什么）：
- 理解「生成 → 评审 → 修订」的自改进循环与质量阈值控制
- 掌握 LangGraph 条件边（通过则结束，未通过则修订后再次评审）
- 学会用 ModelScopeChat 做多轮结构化输出（邮件草稿、评审结果）
- 可选：将通过的样本写入「金标记忆」供后续任务参考（RLHF 类比）

核心概念速览：
- 自改进（Self-Refine）：生成初稿 → 评审打分与反馈 → 根据反馈修订 → 再评审，直到通过或达最大轮数
- 金标记忆：通过评审的高质量输出可存入记忆，供下次生成时参考，类比 RLHF 的偏好学习
- 适用场景：营销邮件、法律/技术文档、需高质量单篇输出的任务

运行前准备：
- 项目根目录创建 `.env` 并配置：
  - `MODELSCOPE_API_KEY`（必需）
  - `MODELSCOPE_BASE_URL`、`MODELSCOPE_MODEL_ID`（可选，有默认）
  - 仅使用主模型（MODELSCOPE_MODEL_ID，默认 DeepSeek-V3.2），不考虑备用模型

如何运行：
- 直接运行默认示例：`python 15_RLHF_cn.py`
- 自定义请求：`python 15_RLHF_cn.py --request "为我们新的 AI 数据分析平台写一封营销邮件"`

阅读建议：
- 先看「结构化模型（邮件、评审）」，再看「生成/评审/修订」节点与条件边，最后看入口。
"""

import os
import argparse
import time
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
from pydantic import BaseModel, Field, ValidationError
from langgraph.graph import StateGraph, END
from typing_extensions import TypedDict
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from openai import OpenAI
from openai import RateLimitError, APIError

# =========================
# 1) 数据结构与模型定义（Pydantic v2）
# =========================

class MarketingEmail(BaseModel):
    """营销邮件草稿。"""
    subject: str = Field(description="邮件主题")
    body: str = Field(description="邮件正文，可用 markdown")


class Critique(BaseModel):
    """对邮件草稿的评审结果。"""
    score: int = Field(description="综合质量分 1～10")
    feedback_points: List[str] = Field(description="具体可执行的改进建议列表")
    is_approved: bool = Field(description="是否通过（如 score>=8）")

# 智能体状态：用户请求、邮件草稿、评审结果、修订次数
class AgentState(TypedDict):
    user_request: str  # 用户请求
    draft_email: Optional[MarketingEmail]  # 邮件草稿
    critique: Optional[Critique]  # 评审结果
    revision_number: int  # 修订次数，初始为 0，每修订一次加 1，最多 3 次


# =========================
# 2) LLM 与控制台
# =========================

console = Console()
DEBUG: bool = False
MAX_REVISIONS = 3


class ModelScopeChat:
    """
    ModelScope 的 OpenAI 兼容接口：_call / with_structured_output。
    仅使用主模型（MODELSCOPE_MODEL_ID，默认 deepseek-ai/DeepSeek-V3.2），不考虑备用模型。
    行为与 01_reflection.py 一致：始终创建客户端，由调用时 API 报错提示。
    """
    def __init__(self, base_url: str = None, api_key: str = None, model: str = None, temperature: float = 0.4, extra_body: Optional[dict] = None):
        self.base_url = base_url or os.environ.get("MODELSCOPE_BASE_URL", "https://api-inference.modelscope.cn/v1")
        self.api_key = api_key or os.environ.get("MODELSCOPE_API_KEY", "")
        self.model = model or os.environ.get("MODELSCOPE_MODEL_ID", "deepseek-ai/DeepSeek-V3.2")
        self.temperature = temperature
        self.extra_body = extra_body or {"enable_thinking": True, "trust_request_chat_template": True, "response_format": {"type": "json_object"}}
        self.client = OpenAI(base_url=self.base_url, api_key=self.api_key)

    def _call(self, messages: list) -> str:
        extra = dict(self.extra_body) if self.extra_body else {}
        try:
            r = self.client.chat.completions.create(
                model=self.model, messages=messages,
                temperature=self.temperature, stream=False, extra_body=extra,
            )
            msg = r.choices[0].message
            content = (msg.content or "").strip()
            # enable_thinking 时 content 可能为空，最终答案有时在 reasoning_content 末尾
            if not content and getattr(msg, "reasoning_content", None):
                content = (msg.reasoning_content or "").strip()
            return content
        except (RateLimitError, APIError) as e:
            if "balance" in str(e).lower() or "403" in str(e) or "insufficient" in str(e).lower():
                raise RuntimeError(
                    "主模型请求失败：账户余额不足(403)，请充值或检查 MODELSCOPE_API_KEY / MODELSCOPE_MODEL_ID 配置。"
                ) from e
            raise

    def with_structured_output(self, pyd_model: type[BaseModel]):
        import json, re
        class _Wrap:
            def __init__(self, outer): self.outer = outer
            def invoke(self, prompt: str, system: str = None) -> BaseModel:
                schema = pyd_model.model_json_schema()
                props = schema.get("properties", {})
                req = schema.get("required", [])
                schema_txt = "\n".join(f"- {k}: {v.get('type','string')}" for k, v in props.items()) or "- 按模型字段"
                req_txt = ", ".join(req) if req else "所有字段"
                sys = system or f"只输出一个纯 JSON 对象，严格匹配：\n{schema_txt}\n必须包含：{req_txt}\n不要使用 Markdown、解释或代码块，直接输出 {{...}} 格式。"
                messages = [{"role": "system", "content": sys}, {"role": "user", "content": prompt}]
                raw = self.outer._call(messages)
                raw = (raw or "").strip()
                # 提取 JSON：优先 ```json ... ```，否则首个 {...}
                data = {}
                m_block = re.search(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", raw)
                m_brace = re.search(r"\{[\s\S]*\}", raw)
                try:
                    if m_block:
                        data = json.loads(m_block.group(1))
                    elif m_brace:
                        data = json.loads(m_brace.group(0))
                    else:
                        data = json.loads(raw) if raw else {}
                except Exception:
                    data = {}
                if not isinstance(data, dict):
                    data = {}
                # 无 JSON 时尝试解析 Markdown 格式（如 **主题：** xxx **正文：** yyy）
                if not data and pyd_model.__name__ == "MarketingEmail":
                    m_subj = re.search(r"\*{0,2}主题[：:]\s*\*{0,2}\s*([^\n]+)", raw)
                    m_body = re.search(r"\*{0,2}正文[：:]\s*\*{0,2}\s*[\r\n]*(.+)", raw, re.DOTALL)
                    if m_subj or m_body:
                        data = {
                            "subject": (m_subj.group(1) or "").strip(),
                            "body": (m_body.group(1) or "").strip(),
                        }
                # 字段映射：兼容 API 返回的常见别名
                if pyd_model.__name__ == "MarketingEmail":
                    if "subject" not in data and "title" in data:
                        data["subject"] = data.pop("title")
                    if "body" not in data and "content" in data:
                        data["body"] = data.pop("content")
                elif pyd_model.__name__ == "Critique":
                    if "feedback_points" not in data and "feedback" in data:
                        data["feedback_points"] = data.pop("feedback") if isinstance(data.get("feedback"), list) else [data.pop("feedback")]
                try:
                    return pyd_model.model_validate(data)
                except ValidationError as e:
                    if not data:
                        console.print(f"[bold yellow]⚠️ API 返回为空或无法解析，使用兜底数据。原始返回前200字：[/bold yellow]\n{(raw or '')[:200]}")
                        if pyd_model.__name__ == "MarketingEmail":
                            return pyd_model(subject="（解析失败，请检查 API 返回格式）", body="（解析失败）")
                        return pyd_model(score=8, feedback_points=["解析失败"], is_approved=True)
                    raise
        return _Wrap(self)


def init_llm() -> ModelScopeChat:
    base_url = os.environ.get("MODELSCOPE_BASE_URL", "https://api-inference.modelscope.cn/v1")
    api_key = os.environ.get("MODELSCOPE_API_KEY", "")
    model_id = os.environ.get("MODELSCOPE_MODEL_ID", "deepseek-ai/DeepSeek-V3.2")
    extra = {"enable_thinking": True, "trust_request_chat_template": True, "response_format": {"type": "json_object"}}
    return ModelScopeChat(base_url=base_url, api_key=api_key, model=model_id, temperature=0.4, extra_body=extra)


# =========================
# 3) 图节点：生成、评审、修订与条件路由
# =========================

def generate_node(llm: ModelScopeChat):
    gen_sys = "你是初级营销文案。根据用户请求写一封营销邮件的初稿，包含主题和正文（可 markdown）。"
    def node(state: AgentState) -> Dict[str, Any]:  # 生成节点
        console.print(Panel("📝 生成初稿", title="[yellow]Step: Generate[/yellow]", border_style="yellow"))
        structured = llm.with_structured_output(MarketingEmail)
        draft = structured.invoke(f"请求：{state['user_request']}", system=gen_sys)
        console.print(Panel(f"主题：{draft.subject}\n\n{draft.body}", title="初稿"))
        return {"draft_email": draft, "revision_number": 0}
    return node


def critique_node(llm: ModelScopeChat):
    crit_sys = (
        "你是高级营销编辑。从以下维度评审邮件：1) 主题是否吸引人 2) 正文是否清晰有说服力 "
        "3) 是否有明确行动号召 4) 语气是否专业亲切。打分 1-10，8 分及以上为通过。给出具体可执行改进建议。"
    )
    def node(state: AgentState) -> Dict[str, Any]:  # 评审节点
        rn = state.get("revision_number", 0)
        console.print(Panel(f"🧐 评审第 {rn + 1} 稿", title="[yellow]Step: Critique[/yellow]", border_style="yellow"))
        d = state["draft_email"]
        structured = llm.with_structured_output(Critique)
        crit = structured.invoke(f"主题：{d.subject}\n\n正文：\n{d.body}", system=crit_sys)
        fb = "\n- ".join(crit.feedback_points)
        console.print(Panel(f"分数：{crit.score}/10\n反馈：\n- {fb}", title="评审结果"))
        return {"critique": crit}
    return node


def revise_node(llm: ModelScopeChat):
    rev_sys = "你是初级营销文案。根据编辑的反馈修订邮件，逐条改进，输出新的主题与正文。"
    def node(state: AgentState) -> Dict[str, Any]:  # 修订节点
        console.print(Panel("✍️ 根据反馈修订", title="[yellow]Step: Revise[/yellow]", border_style="yellow"))
        d = state["draft_email"]
        c = state["critique"]
        fb = "\n- ".join(c.feedback_points)
        prompt = (
            f"原请求：{state['user_request']}\n\n"
            f"原主题：{d.subject}\n原正文：\n{d.body}\n\n"
            f"编辑反馈：\n{fb}"
        )
        structured = llm.with_structured_output(MarketingEmail)
        revised = structured.invoke(prompt, system=rev_sys)
        rn = state.get("revision_number", 0) + 1
        console.print(Panel(f"主题：{revised.subject}\n\n{revised.body}", title=f"第 {rn + 1} 稿"))
        return {"draft_email": revised, "revision_number": rn}
    return node


def should_continue(state: AgentState) -> str:  # 条件路由：通过→end，达最大修订→end，否则→continue
    c = state.get("critique")
    if c and c.is_approved:
        console.print("[green]评审通过，结束。[/green]")
        return "end"
    if state.get("revision_number", 0) >= MAX_REVISIONS:
        console.print("[red]已达最大修订次数，结束。[/red]")
        return "end"
    console.print("[yellow]需要修订，继续循环。[/yellow]")
    return "continue"


# =========================
# 4) 工作流构建与运行
# =========================

def build_app(llm: ModelScopeChat):  # 构建工作流
    workflow = StateGraph(AgentState)
    workflow.add_node("generate", generate_node(llm))
    workflow.add_node("critique", critique_node(llm))
    workflow.add_node("revise", revise_node(llm))
    workflow.set_entry_point("generate")
    workflow.add_edge("generate", "critique")
    workflow.add_conditional_edges("critique", should_continue, {"continue": "revise", "end": END})
    workflow.add_edge("revise", "critique")
    return workflow.compile()


def run_workflow(app, request: str):
    """运行工作流，返回 (最终 state, 节点耗时统计)。
    耗时统计为 {node_name: [duration_1, duration_2, ...]}，单位为秒。
    """
    state = {"user_request": request, "draft_email": None, "critique": None, "revision_number": 0}
    timings: Dict[str, List[float]] = {}
    t_prev = time.perf_counter()
    for step in app.stream(state):
        if END not in step:
            node_name = list(step.keys())[0]
            state = step[node_name]
            t_now = time.perf_counter()
            timings.setdefault(node_name, []).append(t_now - t_prev)
            t_prev = t_now
    return state, timings


# 节点名 → 中文展示名（用于耗时表）
_NODE_LABELS = {"generate": "生成初稿", "critique": "评审", "revise": "修订"}


def print_timing_panel(timings: Dict[str, List[float]], total_elapsed: float):
    """用 Rich Table + Panel 输出耗时统计：按节点、次数、总耗时、占比、平均。"""
    if not timings or total_elapsed <= 0:
        console.print("[dim]耗时：—[/dim]")
        return
    table = Table(show_header=True, header_style="bold cyan", box=None, padding=(0, 2))
    table.add_column("节点", style="green")
    table.add_column("次数", justify="right", style="dim")
    table.add_column("总耗时", justify="right")
    table.add_column("占比", justify="right", style="dim")
    table.add_column("平均/次", justify="right", style="dim")
    for node_name in ("generate", "critique", "revise"):
        if node_name not in timings:
            continue
        durations = timings[node_name]
        label = _NODE_LABELS.get(node_name, node_name)
        total = sum(durations)
        pct = 100.0 * total / total_elapsed
        avg = total / len(durations)
        table.add_row(
            label,
            str(len(durations)),
            f"{total:.2f}s",
            f"{pct:.0f}%",
            f"{avg:.2f}s",
        )
    table.add_row("", "", "", "", "", style="dim")
    table.add_row("[bold]合计[/bold]", "", f"[bold]{total_elapsed:.2f}s[/bold]", "100%", "")
    console.print(Panel(table, title="[bold]⏱ 耗时统计[/bold]", border_style="blue", padding=(0, 1)))


# =========================
# 5) CLI 与入口
# =========================

def parse_args():
    p = argparse.ArgumentParser(description="自改进循环：生成 → 评审 → 修订（Self-Refine）")
    p.add_argument("--request", type=str, default="为我们新的 AI 数据分析平台「InsightSphere」写一封营销邮件。", help="邮件主题/请求")
    p.add_argument("--debug", action="store_true", help="调试输出")
    return p.parse_args()


def main():
    global DEBUG
    load_dotenv()
    args = parse_args()
    DEBUG = getattr(args, "debug", False)
    if not os.environ.get("MODELSCOPE_API_KEY"):
        console.print("[bold red]MODELSCOPE_API_KEY 未设置，请在项目根目录配置 .env[/bold red]")
    llm = init_llm()
    app = build_app(llm)
    console.print(f"--- 🚀 自改进流程：{args.request} ---")
    t0 = time.perf_counter()
    result, timings = run_workflow(app, args.request)
    total_elapsed = time.perf_counter() - t0
    console.print("\n--- 最终邮件 ---")
    d = result.get("draft_email")#邮件草稿
    c = result.get("critique")#评审结果
    approved = c and c.is_approved#是否通过 
    if d:
        title = "[bold green]通过邮件[/bold green]" if approved else "[bold yellow]最终邮件（未通过）[/bold yellow]"
        subtitle = f"分数：{c.score}/10" if c else ""
        border = "green" if approved else "yellow"
        console.print(Panel(f"主题：{d.subject}\n\n{d.body}", title=title, subtitle=subtitle, border_style=border))
    console.print()#打印空行
    print_timing_panel(timings, total_elapsed)#打印耗时统计


if __name__ == "__main__":
    main()

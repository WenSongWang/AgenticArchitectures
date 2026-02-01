# -*- coding: utf-8 -*-
"""
可观测与试跑外壳（Observability + Dry-Run Harness）架构的可运行示例

学习目标（看完你能做到什么）：
- 理解「先试跑、再人工审核、再真实执行」的安全发布流程
- 掌握带 dry_run 标志的工具与 LangGraph 条件边（审核通过/拒绝）
- 学会用 ModelScopeChat 做结构化输出（拟发布内容）与人机审核交互
- 能运行企业社交媒体拟发帖演示：生成 → 试跑预览 → 输入 approve/reject → 执行或取消

核心概念速览：
- 试跑（Dry Run）：工具在 dry_run=True 下只输出「将要执行的动作」与日志，不产生真实副作用
- 人机审核：将试跑结果展示给操作员，只有输入 approve 才执行真实动作
- 适用场景：发帖、发邮件、改数据库等不可逆操作的前置校验

运行前准备：
- 项目根目录创建 `.env` 并配置：
  - `MODELSCOPE_API_KEY`（必需）
  - `MODELSCOPE_BASE_URL`、`MODELSCOPE_MODEL_ID`（可选，有默认）
  - 仅使用主模型（MODELSCOPE_MODEL_ID，默认 DeepSeek-V3.2），不考虑备用模型

如何运行：
- 直接运行默认示例：`python 14_dry_run_cn.py`
- 自定义请求：`python 14_dry_run_cn.py --request "为我们的新 AI 产品写一条发布公告"`

阅读建议：
- 先看「状态与结构化模型」「试跑工具」，再看「提议/试跑/审核/执行」节点与条件边。
"""

import os
import argparse
import datetime
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, END
from typing_extensions import TypedDict
from rich.console import Console
from rich.panel import Panel
from openai import OpenAI
from openai import RateLimitError, APIError

# =========================
# 1) 数据结构与模型定义（Pydantic v2）
# =========================

class SocialMediaPost(BaseModel):
    """拟发布的社交媒体帖子（结构化）。"""
    content: str = Field(description="帖子正文")
    hashtags: List[str] = Field(description="标签列表，不含 #")


# 智能体状态，包含用户请求、拟发布帖子、试跑日志、审核决定、最终状态
class AgentState(TypedDict):
    user_request: str  # 用户请求
    proposed_post: Optional[SocialMediaPost]  # 拟发布帖子
    dry_run_log: Optional[str]  # 试跑日志
    review_decision: Optional[str]  # 审核决定
    final_status: str  # 最终状态


def initial_state(request: str) -> Dict[str, Any]:
    """按 AgentState 的 schema 构造初始状态：仅 user_request 有值，其余为占位。"""
    return {
        "user_request": request,
        "proposed_post": None,
        "dry_run_log": None,
        "review_decision": None,
        "final_status": "",
    }


# =========================
# 2) 试跑工具与 LLM
# =========================

console = Console()
DEBUG: bool = False


class SocialMediaAPI:
    """支持 dry_run 的模拟发帖 API：dry_run=True 只打日志不真实发帖。"""
    def publish_post(self, post: SocialMediaPost, dry_run: bool = True) -> Dict[str, Any]:
        ts = datetime.datetime.now().isoformat()
        hashtags_str = " ".join(f"#{h}" for h in post.hashtags)
        full_text = f"{post.content}\n\n{hashtags_str}"
        if dry_run:
            log = f"[试跑] {ts} 将发布以下内容：\n--- 预览 ---\n{full_text}\n--- 结束 ---"
            console.print(Panel(log, title="[yellow]试跑日志[/yellow]", border_style="yellow"))
            return {"status": "DRY_RUN_SUCCESS", "log": log, "proposed_post": full_text}
        log = f"[正式] {ts} 已发布。"
        console.print(Panel(log, title="[green]正式执行日志[/green]", border_style="green"))
        return {"status": "LIVE_SUCCESS", "log": log, "post_id": f"post_{hash(full_text) & 0x7FFFFFFF}"}


social_media_tool = SocialMediaAPI()


class ModelScopeChat:
    """
    ModelScope 的 OpenAI 兼容接口：invoke、with_structured_output。
    仅使用主模型（MODELSCOPE_MODEL_ID，默认 deepseek-ai/DeepSeek-V3.2），不考虑备用模型；无 API 密钥时返回模拟响应。
    """
    def __init__(self, base_url: str = None, api_key: str = None, model: str = None, temperature: float = 0.5, extra_body: Optional[dict] = None):
        self.base_url = base_url or os.environ.get("MODELSCOPE_BASE_URL", "https://api-inference.modelscope.cn/v1")
        self.api_key = api_key or os.environ.get("MODELSCOPE_API_KEY")
        self.model = model or os.environ.get("MODELSCOPE_MODEL_ID", "deepseek-ai/DeepSeek-V3.2")
        self.temperature = temperature
        self.extra_body = extra_body or {"enable_thinking": True, "trust_request_chat_template": True, "response_format": {"type": "json_object"}}
        if not self.api_key:
            console.print("[bold yellow]⚠️ 未设置 MODELSCOPE_API_KEY，将使用模拟响应[/bold yellow]")
            self.client = None
        else:
            self.client = OpenAI(base_url=self.base_url, api_key=self.api_key)

    def invoke(self, prompt: str) -> str:
        if not self.client:
            return "（未配置 API，模拟正文与 #AI #发布）"
        extra = dict(self.extra_body) if self.extra_body else {}
        try:
            r = self.client.chat.completions.create(
                model=self.model, messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature, stream=False, extra_body=extra,
            )
            return (r.choices[0].message.content or "").strip()
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
            def invoke(self, prompt: str) -> BaseModel:
                schema = pyd_model.model_json_schema()
                props = schema.get("properties", {})
                req = schema.get("required", [])
                schema_txt = "\n".join(f"- {k}: {v.get('type','string')}" for k, v in props.items()) or "- 按模型字段"
                req_txt = ", ".join(req) if req else "所有字段"
                system = f"只输出一个 JSON 对象，严格匹配：\n{schema_txt}\n必须包含：{req_txt}\n不要解释或代码块。"
                messages = [{"role": "system", "content": system}, {"role": "user", "content": prompt}]
                if not self.outer.client:
                    return pyd_model(content="（模拟正文）", hashtags=["AI", "发布"])
                try:
                    r = self.outer.client.chat.completions.create(
                        model=self.outer.model, messages=messages,
                        temperature=self.outer.temperature, stream=False, extra_body=self.outer.extra_body,
                    )
                except (RateLimitError, APIError) as e:
                    if "balance" in str(e).lower() or "403" in str(e) or "insufficient" in str(e).lower():
                        raise RuntimeError(
                            "主模型请求失败：账户余额不足(403)，请充值或检查 MODELSCOPE_API_KEY / MODELSCOPE_MODEL_ID 配置。"
                        ) from e
                    raise
                raw = (r.choices[0].message.content or "").strip()
                try:
                    data = json.loads(raw)
                except Exception:
                    m = re.search(r"\{[\s\S]*\}", raw)
                    data = json.loads(m.group(0)) if m else {}
                return pyd_model.model_validate(data)
        return _Wrap(self)


def init_llm() -> ModelScopeChat:
    base_url = os.environ.get("MODELSCOPE_BASE_URL", "https://api-inference.modelscope.cn/v1")
    api_key = os.environ.get("MODELSCOPE_API_KEY", "")
    model_id = os.environ.get("MODELSCOPE_MODEL_ID", "deepseek-ai/DeepSeek-V3.2")
    extra = {"enable_thinking": True, "trust_request_chat_template": True, "response_format": {"type": "json_object"}}
    return ModelScopeChat(base_url=base_url, api_key=api_key, model=model_id, temperature=0.5, extra_body=extra)


# =========================
# 3) 图节点：提议、试跑审核、执行、拒绝
# =========================

def propose_post_node(llm: ModelScopeChat):
    def node(state: AgentState) -> Dict[str, Any]:
        console.print("--- 📝 拟稿中 ---")
        prompt = (
            "你是一家 AI 公司的社交媒体运营。根据用户请求，写一条吸引人的帖子并给出若干标签（仅标签名，不要 #）。\n\n"
            f"请求：{state['user_request']}"
        )
        structured = llm.with_structured_output(SocialMediaPost)
        post = structured.invoke(prompt)
        return {"proposed_post": post}
    return node


def dry_run_review_node(state: AgentState) -> Dict[str, Any]:
    console.print("--- 🧐 试跑并等待人工审核 ---")
    result = social_media_tool.publish_post(state["proposed_post"], dry_run=True)
    console.print(Panel(
        result["proposed_post"],
        title="[bold yellow]请审核：输入 approve 发布，reject 取消[/bold yellow]",
        border_style="yellow",
    ))
    decision = ""
    # 唯一需要人工输入处：循环直到输入 approve/reject（含大小写变体，见 .lower()）
    while decision.lower() not in ("approve", "reject"):
        # 输入 approve 或 reject，如果输入的是 approve 或 reject 的变体，比如 APPROVE 或 REJECT，那么会认为输入不合法，需要继续等待用户输入
        #console.input是读取用户输入，和python 的input一样，但是console.input会显示一个提示符，而input不会显示提示符,用户输入的内容会赋值给decision
        decision = console.input("输入 approve 或 reject：").strip() or ""
    return {"dry_run_log": result["log"], "review_decision": decision.lower()}


def execute_live_post_node(state: AgentState) -> Dict[str, Any]:
    console.print("--- ✅ 已批准，正式执行 ---")
    result = social_media_tool.publish_post(state["proposed_post"], dry_run=False)
    return {"final_status": f"发布成功，ID: {result.get('post_id', '')}"}


def post_rejected_node(state: AgentState) -> Dict[str, Any]:
    console.print("--- ❌ 已拒绝，不执行 ---")
    return {"final_status": "已拒绝，未执行发布。"}


def route_after_review(state: AgentState) -> str:
    return "execute_live" if state.get("review_decision") == "approve" else "reject"


# =========================
# 4) 工作流构建与运行
# =========================

def build_app(llm: ModelScopeChat):
    workflow = StateGraph(AgentState)
    workflow.add_node("propose_post", propose_post_node(llm))
    workflow.add_node("dry_run_review", dry_run_review_node)
    workflow.add_node("execute_live", execute_live_post_node)
    workflow.add_node("reject", post_rejected_node)
    workflow.set_entry_point("propose_post")
    workflow.add_edge("propose_post", "dry_run_review")
    workflow.add_conditional_edges("dry_run_review", route_after_review, {"execute_live": "execute_live", "reject": "reject"})
    workflow.add_edge("execute_live", END)
    workflow.add_edge("reject", END)
    return workflow.compile()


def run_workflow(app, request: str) -> Dict[str, Any]:
    """传入初始状态（由 initial_state 提供），运行图并返回最终状态。"""
    return app.invoke(initial_state(request))


# =========================
# 5) CLI 与入口
# =========================

def parse_args():
    p = argparse.ArgumentParser(description="可观测与试跑外壳：拟发帖 → 试跑 → 审核 → 执行/取消")
    p.add_argument("--request", type=str, default="为我们的新 AI 模型「星云」写一条正面发布公告。", help="发帖请求")
    p.add_argument("--debug", action="store_true", help="调试输出")
    return p.parse_args()


def main():
    global DEBUG
    load_dotenv()
    args = parse_args()
    DEBUG = getattr(args, "debug", False)
    if not os.environ.get("MODELSCOPE_API_KEY"):
        console.print("[bold yellow]未设置 MODELSCOPE_API_KEY，将使用模拟响应[/bold yellow]")
    llm = init_llm()
    app = build_app(llm)
    console.print(f"--- 请求：{args.request} ---")
    result = run_workflow(app, args.request)
    console.print(f"\n[bold]最终状态：[/bold] {result.get('final_status', '')}")


if __name__ == "__main__":
    main()

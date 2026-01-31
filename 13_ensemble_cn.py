# -*- coding: utf-8 -*-
"""
并行探索 + 集成决策（Parallel Exploration + Ensemble Decision）架构的可运行示例

学习目标（看完你能做到什么）：
- 理解多路并行探索、多视角分析后由聚合者综合结论的流程
- 掌握 LangGraph 的扇出/扇入（多节点并行与汇聚）
- 学会用 ModelScopeChat 做纯文本与结构化输出（CIO 综合报告）
- 能运行「投资委员会」演示：三路分析师 + CIO 综合

核心概念速览：
- 并行探索：同一问题由多个独立智能体（不同人设）同时分析，得到多份报告
- 集成决策：聚合智能体（如 CIO）综合多份报告，产出最终结构化结论
- 适用场景：复杂推理、事实核查、高 stakes 决策支持

运行前准备：
- 项目根目录创建 `.env` 并配置：
  - `MODELSCOPE_API_KEY`（必需）
  - `MODELSCOPE_BASE_URL`、`MODELSCOPE_MODEL_ID`（可选，有默认）
  - 仅使用主模型（MODELSCOPE_MODEL_ID，默认 DeepSeek-V3.2），不考虑备用模型
  - 可选 Tavily：`TAVILY_API_KEY`（分析师检索用；未配置时用模拟检索）

如何运行：
- 直接运行默认示例：`python 13_ensemble_cn.py`
- 自定义问题：`python 13_ensemble_cn.py --request "英伟达 2026 中长期是否值得投资？"`

阅读建议：
- 先看「状态与结构化模型」，再看「分析师节点」与「CIO 聚合节点」，最后看图构建与入口。
"""

import os
import asyncio
import argparse
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, END
from typing_extensions import TypedDict
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from openai import OpenAI
from openai import RateLimitError, APIError

# =========================
# 1) 数据结构与模型定义（Pydantic v2）
# =========================

class FinalRecommendation(BaseModel):
    """CIO 综合后的最终投资建议（结构化）。"""
    final_recommendation: str = Field(description="最终投资结论，须为 Strong Buy / Buy / Hold / Sell / Strong Sell 之一")
    confidence_score: float = Field(description="信心分数，1.0～10.0")
    synthesis_summary: str = Field(description="综合各分析师观点的摘要，含共识与分歧")
    identified_opportunities: List[str] = Field(description="主要机会或看多要点（列表）")
    identified_risks: List[str] = Field(description="主要风险或看空要点（列表）")


class EnsembleState(TypedDict):
    query: str
    analyses: Dict[str, str]
    final_recommendation: Optional[Any]


# =========================
# 2) LLM 与控制台
# =========================

console = Console()
DEBUG: bool = False


def _search_or_mock(query: str) -> str:
    """Tavily 检索或模拟检索结果。"""
    if os.environ.get("TAVILY_API_KEY"):
        try:
            from langchain_tavily import TavilySearch
            search = TavilySearch(max_results=3)
            docs = search.invoke(query)
            return docs if isinstance(docs, str) else "\n\n".join(getattr(d, "content", str(d)) for d in (docs if isinstance(docs, list) else [docs]))
        except Exception as e:
            if DEBUG:
                console.print(f"[dim]Tavily 检索异常: {e}[/dim]")
    return f"[模拟检索] 与「{query[:50]}」相关的近期市场与基本面摘要（未配置 TAVILY_API_KEY 时使用）。"


class ModelScopeChat:
    """
    ModelScope 的 OpenAI 兼容接口：invoke、with_structured_output。
    仅使用主模型（MODELSCOPE_MODEL_ID，默认 deepseek-ai/DeepSeek-V3.2），不考虑备用模型；无 API 密钥时返回模拟响应。
    """
    def __init__(self, base_url: str = None, api_key: str = None, model: str = None, temperature: float = 0.3, extra_body: Optional[dict] = None):
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
            return "（未配置 API，此为模拟分析文本。）"
        extra = dict(self.extra_body) if self.extra_body else {}
        try:
            resp = self.client.chat.completions.create(
                model=self.model, messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature, stream=False, extra_body=extra,
            )
            return (resp.choices[0].message.content or "").strip()
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
                schema_txt = "\n".join(f"- {k}: {v.get('type','string')}" for k,v in props.items()) or "- 按模型字段"
                req_txt = ", ".join(req) if req else "所有字段"
                system = f"你只输出一个 JSON 对象，严格匹配以下字段与类型：\n{schema_txt}\n必须包含：{req_txt}\n不要输出解释或代码块。"
                messages = [{"role": "system", "content": system}, {"role": "user", "content": prompt}]
                if not self.outer.client:
                    return pyd_model(final_recommendation="Hold", confidence_score=5.0, synthesis_summary="（模拟）", identified_opportunities=[], identified_risks=[])
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
    return ModelScopeChat(base_url=base_url, api_key=api_key, model=model_id, temperature=0.3, extra_body=extra)


# =========================
# 3) 分析师节点（并行）与 CIO 聚合节点
# =========================

def _run_one_analyst(llm: ModelScopeChat, persona: str, agent_name: str, query: str) -> str:
    """单路分析师：检索 + LLM 分析，返回分析文本。"""
    console.print(f"--- 👨‍💻 调用 {agent_name} ---")
    search_result = _search_or_mock(query)
    prompt = (
        f"你是一名专业金融分析师。你的人设：{persona}\n"
        f"请结合以下研究摘要进行分析：\n{search_result}\n\n"
        f"用户问题：{query}\n\n"
        "请给出详细投资分析，并在文末明确写出「建议」（Buy/Hold/Sell）和「信心分数」（1-10）。"
    )
    return llm.invoke(prompt)


def run_all_analysts_node(llm: ModelScopeChat):
    """并行逻辑：用 asyncio.to_thread 在协程中并发调用三位分析师（同步 LLM 不变）。"""
    personas = [
        ("看多成长分析师：关注 TAM、技术护城河与长期增长，淡化短期估值。", "BullishAnalyst"),
        ("谨慎价值分析师：关注财报、估值、债务与竞争风险，警惕泡沫。", "ValueAnalyst"),
        ("量化分析师：仅基于数据与指标（收入增速、EPS、估值倍数、技术指标）做客观分析。", "QuantAnalyst"),
    ]

    async def node(state: EnsembleState) -> Dict[str, Any]:
        query = state["query"]
        names = [name for _, name in personas]
        tasks = [
            asyncio.to_thread(_run_one_analyst, llm, persona, name, query)
            for persona, name in personas
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        analyses = {}
        for name, r in zip(names, results):
            analyses[name] = r if not isinstance(r, BaseException) else f"[分析异常] {r}"
        return {"analyses": analyses}

    return node


def start_analysis_node(state: EnsembleState) -> Dict[str, Any]:
    return {"analyses": {}}


def cio_synthesizer_node(llm: ModelScopeChat):
    def node(state: EnsembleState) -> Dict[str, Any]:
        console.print("--- 🏛️ 调用首席投资官（CIO）综合决策 ---")
        all_analyses = "\n\n---\n\n".join(
            f"**{name} 分析：**\n{text}" for name, text in state["analyses"].items()
        )
        prompt = (
            f"你是一家投资基金的首席投资官（CIO）。以下是团队对同一问题的多份分析报告。\n"
            f"用户问题：{state['query']}\n\n"
            f"团队报告：\n{all_analyses}\n\n"
            "请综合以上观点，给出最终投资建议（Strong Buy/Buy/Hold/Sell/Strong Sell）、信心分数(1-10)、综合摘要、主要机会与主要风险列表。"
        )
        structured_llm = llm.with_structured_output(FinalRecommendation)
        final_rec = structured_llm.invoke(prompt)
        return {"final_recommendation": final_rec}
    return node


# =========================
# 4) 工作流构建与运行
# =========================

def build_app(llm: ModelScopeChat):
    workflow = StateGraph(EnsembleState)
    workflow.add_node("start_analysis", start_analysis_node)
    workflow.add_node("run_all_analysts", run_all_analysts_node(llm))
    workflow.add_node("cio_synthesizer", cio_synthesizer_node(llm))
    workflow.set_entry_point("start_analysis")
    workflow.add_edge("start_analysis", "run_all_analysts")
    workflow.add_edge("run_all_analysts", "cio_synthesizer")
    workflow.add_edge("cio_synthesizer", END)
    return workflow.compile()


async def run_workflow_async(app, request: str) -> Dict[str, Any]:
    return await app.ainvoke({"query": request, "analyses": {}, "final_recommendation": None})


def run_workflow(app, request: str) -> Dict[str, Any]:
    return asyncio.run(run_workflow_async(app, request))


# =========================
# 5) CLI 与入口
# =========================

def parse_args():
    p = argparse.ArgumentParser(description="并行探索 + 集成决策：投资委员会演示")
    p.add_argument("--request", type=str, default="基于近期新闻、财务表现与展望，英伟达（NVDA）在 2026 年下半年是否值得长期投资？", help="投资分析问题")
    p.add_argument("--debug", action="store_true", help="开启调试输出")
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
    console.print(f"--- 📈 投资委员会分析：{args.request} ---")
    result = run_workflow(app, args.request)
    console.print("\n--- 各分析师报告 ---")
    for name, analysis in result.get("analyses", {}).items():
        console.print(Panel(Markdown(analysis), title=f"[bold yellow]{name}[/bold yellow]", border_style="yellow"))
    rec = result.get("final_recommendation")
    if rec:
        console.print("\n--- CIO 综合建议 ---")
        console.print(Panel(
            f"[bold]最终建议：[/bold] {rec.final_recommendation}\n"
            f"[bold]信心分数：[/bold] {rec.confidence_score}/10\n\n"
            f"[bold]综合摘要：[/bold]\n{rec.synthesis_summary}\n\n"
            f"[bold]主要机会：[/bold]\n" + "\n".join(f"* {x}" for x in rec.identified_opportunities) + "\n\n"
            f"[bold]主要风险：[/bold]\n" + "\n".join(f"* {x}" for x in rec.identified_risks),
            title="[bold green]首席投资官结论[/bold green]", border_style="green"
        ))


if __name__ == "__main__":
    main()

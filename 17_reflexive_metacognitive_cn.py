# -*- coding: utf-8 -*-
"""
反思式元认知智能体（Reflexive Metacognitive Agent）架构的可运行示例

学习目标（看完你能做到什么）：
- 理解「先对请求做元认知分析、再选策略」的安全决策流程
- 掌握基于自模型（知识域、工具、置信度阈值）的策略路由：直接回答 / 用工具 / 升级人工
- 能运行
    - 医疗分诊演示：简单问题直接答、药物相互作用用工具、高危或超域则升级
    - 车舱智能助手演示：简单问题直接答、车舱设备、娱乐、poi、天气、导航、音乐、电话、视频、游戏、新闻、设置、帮助用工具、高危或超域(如车舱设备故障,高危车控操作)则升级

核心概念速览：
- 元认知分析：在回答前先分析「我能否安全且准确回答」「是否需工具」「是否应升级」
- 自模型：智能体对自身知识域、可用工具、置信度阈值的显式描述
- 适用场景：医疗/法律/金融等高风险领域，必须能说「请咨询专业人士」

运行前准备：
- 项目根目录创建 `.env` 并配置：
  - `MODELSCOPE_API_KEY`（必需）
  - `MODELSCOPE_BASE_URL`、`MODELSCOPE_MODEL_ID`（可选，有默认）
  - 仅使用主模型（MODELSCOPE_MODEL_ID，默认 DeepSeek-V3.2），不考虑备用模型

如何运行：
- 直接运行（默认触发 use_tool 药物查询）：`python 17_reflexive_metacognitive_cn.py`
- 直接回答示例：`python 17_reflexive_metacognitive_cn.py --request "感冒和流感的症状有什么区别？"`
- 升级人工示例：`python 17_reflexive_metacognitive_cn.py --request "我胸口疼呼吸困难怎么办？"`

阅读建议：
- 先看「自模型与工具」「元认知分析结构」，再看各策略节点与条件边。
"""

import os
import argparse
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

class AgentSelfModel(BaseModel):
    """智能体自模型：名称、角色、知识域、可用工具、置信度阈值。"""
    name: str = Field(description="智能体名称")
    role: str = Field(description="角色描述")
    knowledge_domain: List[str] = Field(description="擅长的知识域列表")
    available_tools: List[str] = Field(description="可用工具名列表")
    confidence_threshold: float = Field(default=0.6, description="低于此置信度须升级")


class MetacognitiveAnalysis(BaseModel):
    """元认知分析结果。"""
    confidence: float = Field(description="置信度 0.0～1.0")
    strategy: str = Field(description="策略：reason_directly / use_tool / escalate")#reason_directly: 直接回答，use_tool: 使用工具，escalate: 升级人工
    reasoning: str = Field(description="选择该策略的理由")#选择该策略的理由
    tool_to_use: Optional[str] = Field(default=None, description="若 use_tool，工具名")#工具名
    tool_args: Optional[Dict[str, Any]] = Field(default=None, description="若 use_tool，工具参数")#工具参数


class AgentState(TypedDict):
    user_query: str#用户查询
    self_model: AgentSelfModel#智能体自模型
    metacognitive_analysis: Optional[MetacognitiveAnalysis]#元认知分析结果
    tool_output: Optional[str]#工具输出
    final_response: str#最终回复


# =========================
# 2) 工具与 LLM
# =========================

console = Console()
DEBUG: bool = False


class DrugInteractionChecker:
    """模拟药物相互作用查询工具。"""
    def check(self, drug_a: str, drug_b: str) -> str:
        """查询药物相互作用。"""
        known = {
            frozenset(["ibuprofen", "lisinopril"]): "中等风险：布洛芬可能减弱赖诺普利降压效果，需监测血压。",
            frozenset(["aspirin", "warfarin"]): "高风险：增加出血风险，应避免联用，除非医生指导。",
        }
        key = frozenset([drug_a.lower().strip(), drug_b.lower().strip()])
        return known.get(key, "未发现已知显著相互作用。但仍请咨询药师或医生。")


drug_tool = DrugInteractionChecker()


class ModelScopeChat:
    """
j    ModelScope 的 OpenAI 兼容接口：invoke、with_structured_output。
    仅使用主模型（MODELSCOPE_MODEL_ID，默认 deepseek-ai/DeepSeek-V3.2），不考虑备用模型；无 API 密钥时返回模拟响应。
    """
    def __init__(self, base_url: str = None, api_key: str = None, model: str = None, temperature: float = 0, extra_body: Optional[dict] = None):
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
            return "（未配置 API，此为模拟回复。请咨询医生。）"
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
                    return pyd_model(confidence=0.9, strategy="reason_directly", reasoning="模拟", tool_to_use=None, tool_args=None)
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
                # LLM 有时返回 tool_args="", tool_to_use=""，需清洗为 None / 有效 dict
                if "tool_args" in data and (not isinstance(data["tool_args"], dict) or data["tool_args"] == ""):
                    data["tool_args"] = None
                if "tool_to_use" in data and data["tool_to_use"] == "":
                    data["tool_to_use"] = None
                return pyd_model.model_validate(data)
        return _Wrap(self)


def init_llm() -> ModelScopeChat:
    base_url = os.environ.get("MODELSCOPE_BASE_URL", "https://api-inference.modelscope.cn/v1")
    api_key = os.environ.get("MODELSCOPE_API_KEY", "")
    model_id = os.environ.get("MODELSCOPE_MODEL_ID", "deepseek-ai/DeepSeek-V3.2")
    extra = {"enable_thinking": True, "trust_request_chat_template": True, "response_format": {"type": "json_object"}}
    return ModelScopeChat(base_url=base_url, api_key=api_key, model=model_id, temperature=0, extra_body=extra)


# =========================
# 3) 图节点：元认知分析、直接回答、用工具、综合、升级;
#    其中元认知分析用于分析用户查询，选择最安全、最合适的策略。直接回答用于直接回答用户查询。用工具用于使用工具。综合用于综合工具结果并回复。升级用于升级人工。
# =========================

def metacognitive_analysis_node(llm: ModelScopeChat):
    """元认知分析节点。"""
    def node(state: AgentState) -> Dict[str, Any]:
        console.print(Panel("🤔 元认知分析中…", title="[yellow]Step: Self-Reflection[/yellow]", border_style="yellow"))
        sm = state["self_model"]
        """构建元认知分析提示。"""
        prompt = (
            "你是元认知推理引擎。根据智能体自模型分析用户查询，选择最安全、最合适的策略。\n\n"
            f"自模型：名称={sm.name}，角色={sm.role}，知识域={sm.knowledge_domain}，可用工具={sm.available_tools}。\n\n"
            "策略规则：1) escalate：涉及急症、超出知识域或任何不确定时选此项。2) use_tool：问题明确需要某工具时（如药物相互作用用 drug_interaction_checker）。"
            "3) reason_directly：仅当高置信、低风险且完全在知识域内时选此项。\n\n"
            f"用户查询：\"{state['user_query']}\"\n\n"
            "输出 JSON：confidence(0-1), strategy(三者之一), reasoning, tool_to_use(可选), tool_args(可选，如 {\"drug_a\":\"x\",\"drug_b\":\"y\"})。"
        )
        structured = llm.with_structured_output(MetacognitiveAnalysis)#使用结构化输出获取元认知分析结果
        analysis = structured.invoke(prompt)
        console.print(Panel(f"置信度：{analysis.confidence:.2f}\n策略：{analysis.strategy}\n理由：{analysis.reasoning}", title="元认知结果"))
        return {"metacognitive_analysis": analysis}
    return node


def reason_directly_node(llm: ModelScopeChat):
    """直接回答节点。"""
    def node(state: AgentState) -> Dict[str, Any]:
        console.print(Panel("✅ 策略：直接回答", title="[green]Reason Directly[/green]", border_style="green"))
        """构建直接回答提示。"""
        prompt = f"你是{state['self_model'].role}。请对以下问题给出有帮助、不具处方性的回答，并提醒用户你不是医生。\n\n问题：{state['user_query']}"
        resp = llm.invoke(prompt)#使用LLM直接回答用户查询
        return {"final_response": resp}
    return node


def call_tool_node(state: AgentState) -> Dict[str, Any]:
    """使用工具节点。"""
    console.print(Panel("🛠️ 策略：使用工具", title="[cyan]Use Tool[/cyan]", border_style="cyan"))
    analysis = state["metacognitive_analysis"]#获取元认知分析结果
    if analysis.tool_to_use == "drug_interaction_checker" and analysis.tool_args:
        out = drug_tool.check(#使用药物相互作用查询工具查询药物相互作用 
            analysis.tool_args.get("drug_a", ""),
            analysis.tool_args.get("drug_b", ""),
        )
        return {"tool_output": out}
    return {"tool_output": "工具未找到或参数缺失。"}


def synthesize_tool_response_node(llm: ModelScopeChat):
    """综合工具结果并回复节点。"""
    def node(state: AgentState) -> Dict[str, Any]:
        console.print(Panel("📝 综合工具结果并回复", title="[cyan]Synthesize[/cyan]", border_style="cyan"))
        """构建综合工具结果并回复提示。"""
        prompt = (
            f"你是{state['self_model'].role}。你已通过工具获得信息，请清晰、有帮助地呈现给用户，并务必提醒其咨询医疗专业人士。\n\n"
            f"原问题：{state['user_query']}\n工具输出：{state['tool_output']}"
        )
        resp = llm.invoke(prompt)#使用LLM综合工具结果并回复
        return {"final_response": resp}
    return node


def escalate_to_human_node(state: AgentState) -> Dict[str, Any]:
    """升级人工节点。"""
    console.print(Panel("🚨 策略：升级人工", title="[bold red]Escalate[/bold red]", border_style="red"))
    return {"final_response": "我是 AI 助手，无法就此类问题提供专业意见。该问题超出我的知识范围或涉及可能严重症状，请立即咨询合格医疗人员。"}


def route_strategy(state: AgentState) -> str:
    """路由策略节点。"""
    return state["metacognitive_analysis"].strategy#返回元认知分析结果的策略


# =========================
# 4) 工作流构建与运行
# =========================

def build_app(llm: ModelScopeChat):
    """构建工作流。"""
    workflow = StateGraph(AgentState)#创建状态图
    workflow.add_node("analyze", metacognitive_analysis_node(llm))#添加元认知分析节点
    workflow.add_node("reason", reason_directly_node(llm))#添加直接回答节点
    workflow.add_node("call_tool", call_tool_node)#添加使用工具节点
    workflow.add_node("synthesize", synthesize_tool_response_node(llm))#添加综合工具结果并回复节点
    workflow.add_node("escalate", escalate_to_human_node)#添加升级人工节点
    workflow.set_entry_point("analyze")#设置入口点
    workflow.add_conditional_edges("analyze", route_strategy, {#添加条件边：元认知分析 → 直接回答 / 使用工具 / 升级人工
        "reason_directly": "reason",
        "use_tool": "call_tool",
        "escalate": "escalate",
    })
    workflow.add_edge("call_tool", "synthesize")#添加边：使用工具 → 综合工具结果并回复：使用工具 → 综合工具结果并回复
    workflow.add_edge("reason", END)#添加边：直接回答 → 结束：直接回答 → 结束
    workflow.add_edge("synthesize", END)#添加边：综合工具结果并回复 → 结束：综合工具结果并回复 → 结束
    workflow.add_edge("escalate", END)#添加边：升级人工 → 结束：升级人工 → 结束   
    return workflow.compile()#编译工作流







def run_agent(app, query: str, self_model: AgentSelfModel) -> Dict[str, Any]:
    """运行智能体。"""
    return app.invoke({#调用工作流，query 为用户查询，self_model 为智能体自模型
        "user_query": query,
        "self_model": self_model,#默认医疗分诊智能体自模型
        "metacognitive_analysis": None,#默认元认知分析结果
        "tool_output": None,#默认工具输出
        "final_response": "",#默认最终回复
    })


# =========================
# 5) CLI 与入口
# =========================

MEDICAL_SELF_MODEL = AgentSelfModel(#医疗分诊智能体自模型
    name="TriageBot-3000",
    role="提供初步医疗信息的 AI 助手",
    knowledge_domain=["感冒", "流感", "过敏", "头痛", "基础急救"],
    available_tools=["drug_interaction_checker"],
)


def parse_args():
    """解析命令行参数。"""
    p = argparse.ArgumentParser(description="反思式元认知智能体：医疗分诊演示")
    p.add_argument("--request", type=str, default="布洛芬和赖诺普利能一起吃吗？", help="用户问题（默认触发 use_tool 药物查询）")
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
    console.print(f"--- 用户问题：{args.request} ---")
    result = run_agent(app, args.request, MEDICAL_SELF_MODEL)#运行智能体，args.request 为用户查询，MEDICAL_SELF_MODEL 为医疗分诊智能体自模型
    console.print("\n--- 最终回复 ---")
    console.print(Panel(result.get("final_response", ""), title="[bold green]回复[/bold green]", border_style="green"))


if __name__ == "__main__":
    main()

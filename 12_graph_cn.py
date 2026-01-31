# -*- coding: utf-8 -*-
"""
图/世界模型记忆（Graph / World-Model Memory）架构的可运行示例

学习目标（看完你能做到什么）：
- 理解以图结构存储知识：实体为节点、关系为边，构成可查询的世界模型
- 掌握从非结构化文本中抽取实体与关系（LLM + 结构化输出）并写入图
- 学会「自然语言 → Cypher 查询 → 图执行 → 自然语言回答」的 Text-to-Cypher 流程
- 能运行企业情报演示：摄入文档、构建图、回答多跳推理问题

核心概念速览：
- 知识图谱：节点（实体，如 Company、Person）与边（关系，如 WORKS_FOR、ACQUIRED）
- 图构建智能体：从文本中抽取 KnowledgeGraph（Pydantic），再写入图存储
- Text-to-Cypher：用户问题 → LLM 生成 Cypher → 图执行 → LLM 合成答案
- 多跳推理：需连接多步关系的问题（如「收购 BetaSolutions 的公司所生产产品的竞品公司」）

运行前准备：
- 项目根目录创建 `.env` 并配置：
  - `MODELSCOPE_API_KEY`（必需，用于 LLM 抽取与生成）
  - `MODELSCOPE_BASE_URL`、`MODELSCOPE_MODEL_ID`（可选，有默认）
  - 本脚本仅使用主模型（MODELSCOPE_MODEL_ID，默认 DeepSeek-V3.2），不考虑备用模型
  - 可选 Neo4j：`NEO4J_URI`、`NEO4J_USERNAME`、`NEO4J_PASSWORD`（未配置时使用内存图演示）

如何运行：
- 直接运行默认示例：`python 12_graph_cn.py`
- 自定义问题：`python 12_graph_cn.py --question "谁在 AlphaCorp 工作？"`
- 仅构建图不查询：`python 12_graph_cn.py --ingest-only`

阅读建议：
- 先看「数据结构与模型」和「图存储」，再看「图构建智能体」与「图查询智能体」的调用方式。
"""

import os
import re
from typing import List, Dict, Any, Optional

from dotenv import load_dotenv
from pydantic import BaseModel, Field

from rich.console import Console
from rich.markdown import Markdown

from openai import OpenAI
from openai import RateLimitError, APIError, PermissionDeniedError

# =========================
# 1) 数据结构与模型定义（Pydantic v2）
# =========================
# 知识图谱在代码中用「节点 + 关系」表示：Node 为实体，Relationship 为有向边，
# 三者组合成 KnowledgeGraph，供 LLM 结构化抽取与图存储写入使用。


class Node(BaseModel):
    """
    实体节点：知识图谱中的一个实体。
    - id：实体的唯一名称（如公司名、人名、产品名），作为图中节点的标识
    - type：实体类型标签，用于 Cypher 查询中的节点标签（如 Person, Company, Product）
    """
    id: str = Field(description="实体的唯一名称或标识")
    type: str = Field(description="实体类型，如 Person, Company, Product")


class Relationship(BaseModel):
    """
    关系边：连接两个实体的一条有向关系。
    - source：关系的起点实体（节点）
    - target：关系的终点实体（节点）
    - type：关系类型，约定为全大写英文动词（如 WORKS_FOR, ACQUIRED），便于 Cypher 与图查询
    """
    source: Node = Field(description="源实体")
    target: Node = Field(description="目标实体")
    type: str = Field(description="关系类型，全大写动词，如 WORKS_FOR, ACQUIRED")


class KnowledgeGraph(BaseModel):
    """
    从单段文本中抽取出的知识图谱。
    - relationships：所有关系列表；每条关系内已包含 source/target 节点，无需单独存节点列表。
    """
    relationships: List[Relationship] = Field(description="所有关系（含节点）")


# =========================
# 2) LLM 与控制台初始化
# =========================
# 使用项目统一的 ModelScopeChat：通过 .env 配置 ModelScope API，支持 invoke 与
# with_structured_output；与 LangChain 的 prompt|llm 不兼容，需用 prompt.format + llm.invoke。

console = Console()
DEBUG: bool = False  # 为 True 时打印更多中间结果（如生成的 Cypher、解析过程）


class ModelScopeChat:
    """
    ModelScope 的 OpenAI 兼容接口适配器：
    - invoke(prompt)：传入一段提示，返回模型生成的纯文本（用于 Cypher 生成、答案合成等）
    - with_structured_output(PydanticModel)：返回一个包装器，invoke(prompt) 得到 Pydantic 实例（用于知识图谱抽取）
    - 仅使用主模型（MODELSCOPE_MODEL_ID，默认 deepseek-ai/DeepSeek-V3.2），不考虑备用模型
    """

    def __init__(
        self,
        base_url: str = None,
        api_key: str = None,
        model: str = None,
        temperature: float = 0.2,
        extra_body: Optional[dict] = None,
    ):
        self.base_url = base_url or os.environ.get(
            "MODELSCOPE_BASE_URL", "https://api-inference.modelscope.cn/v1"
        )
        self.api_key = api_key or os.environ.get("MODELSCOPE_API_KEY")
        self.model = model or os.environ.get(
            "MODELSCOPE_MODEL_ID", "deepseek-ai/DeepSeek-V3.2"
        )
        self.temperature = temperature
        self.extra_body = extra_body or {
            "enable_thinking": True,
            "trust_request_chat_template": True,
            "response_format": {"type": "json_object"},
        }
        if not self.api_key:
            console.print(
                "[bold yellow]⚠️ 未设置 MODELSCOPE_API_KEY，将使用模拟响应[/bold yellow]"
            )
            self.client = None
        else:
            self.client = OpenAI(base_url=self.base_url, api_key=self.api_key)

    def invoke(self, prompt: str, use_json_format: bool = False) -> str:
        """
        调用模型返回纯文本。
        use_json_format=False 时不传 response_format，避免 API 要求消息中含「json」导致 400。
        """
        if not self.client:
            return _mock_llm_response(prompt)
        extra = dict(self.extra_body) if self.extra_body else {}
        if not use_json_format:
            extra.pop("response_format", None)
        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                stream=False,
                extra_body=extra,
            )
            return (resp.choices[0].message.content or "").strip()
        except (RateLimitError, APIError, PermissionDeniedError) as e:
            if "balance" in str(e).lower() or "403" in str(e) or "insufficient" in str(e).lower():
                raise RuntimeError(
                    "主模型请求失败：账户余额不足(403)，请充值或检查 MODELSCOPE_API_KEY / MODELSCOPE_MODEL_ID 配置。"
                ) from e
            raise

    def with_structured_output(self, pyd_model: type[BaseModel]):
        """
        返回一个可调用的包装器，对任意 prompt 调用 invoke(prompt) 时：
        会向模型注入「只输出 JSON、严格按 schema」的系统提示，并将返回内容解析为 pyd_model 实例。
        """

        class _Wrapper:
            def __init__(self, outer: "ModelScopeChat"):
                self.outer = outer

            def invoke(self, prompt: str) -> BaseModel:
                # 从 Pydantic 模型生成 JSON schema 描述，拼进系统提示，约束模型输出格式
                schema = pyd_model.model_json_schema()
                props = schema.get("properties", {})
                required = schema.get("required", [])
                lines = [f"- {k}: {v.get('type', 'string')}" for k, v in props.items()]
                schema_txt = "\n".join(lines) or "- 按模型字段输出"
                req_txt = ", ".join(required) if required else "所有字段"
                system = (
                    "你是一个结构化输出生成器。只输出一个 JSON 对象，严格匹配以下字段与类型：\n"
                    f"{schema_txt}\n必须包含：{req_txt}\n不要输出解释或代码块标记。"
                )
                messages = [
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt},
                ]
                if not self.outer.client:
                    return _mock_structured(pyd_model, prompt)
                import json
                try:
                    raw = self.outer.client.chat.completions.create(
                        model=self.outer.model,
                        messages=messages,
                        temperature=self.outer.temperature,
                        stream=False,
                        extra_body=self.outer.extra_body,
                    )
                except (RateLimitError, APIError, PermissionDeniedError) as e:
                    if "balance" in str(e).lower() or "403" in str(e) or "insufficient" in str(e).lower():
                        raise RuntimeError(
                            "主模型请求失败：账户余额不足(403)，请充值或检查 MODELSCOPE_API_KEY / MODELSCOPE_MODEL_ID 配置。"
                        ) from e
                    raise
                content = (raw.choices[0].message.content or "").strip()
                try:
                    data = json.loads(content)
                except Exception:
                    m = re.search(r"\{[\s\S]*\}", content)
                    data = json.loads(m.group(0)) if m else {}
                # 规范化嵌套结构，兼容模型返回的 source/target 为 dict 或字段名不一致
                if pyd_model == KnowledgeGraph and "relationships" in data:
                    data = _normalize_knowledge_graph_data(data)
                return pyd_model.model_validate(data)

        return _Wrapper(self)


def _mock_llm_response(prompt: str) -> str:
    """未配置 MODELSCOPE_API_KEY 时的占位文本：Cypher 类提示返回简单查询，其余返回说明文字。"""
    if "Cypher" in prompt or "cypher" in prompt.lower():
        return "MATCH (n) RETURN n LIMIT 1"
    return "（未配置 API，此为模拟回答。）"


def _normalize_knowledge_graph_data(data: dict) -> dict:
    """
    将模型返回的 JSON 规范化为 KnowledgeGraph 所需格式。
    兼容：source/target 为 dict（字段 id/type 或 name/label）或为字符串（实体名）。
    """
    rels = data.get("relationships") or []
    normalized = []
    for r in rels:
        if not isinstance(r, dict):
            continue
        src = r.get("source") or r.get("from")
        tgt = r.get("target") or r.get("to")
        typ = (r.get("type") or r.get("relationship_type") or "").strip()
        if isinstance(src, dict):
            src_id = (src.get("id") or src.get("name") or str(src)).strip()
            src_type = (src.get("type") or src.get("label") or "Entity").strip()
        else:
            src_id = str(src).strip() if src else ""
            src_type = "Entity"
        if isinstance(tgt, dict):
            tgt_id = (tgt.get("id") or tgt.get("name") or str(tgt)).strip()
            tgt_type = (tgt.get("type") or tgt.get("label") or "Entity").strip()
        else:
            tgt_id = str(tgt).strip() if tgt else ""
            tgt_type = "Entity"
        if not src_id or not tgt_id or not typ:
            continue
        normalized.append({
            "source": {"id": src_id, "type": src_type},
            "target": {"id": tgt_id, "type": tgt_type},
            "type": typ,
        })
    return {"relationships": normalized}


def _mock_knowledge_graph_for_demo(prompt: str) -> KnowledgeGraph:
    """
    未配置 API 时，对默认演示文档返回预置知识图谱，使无 Neo4j/无 API 时内存图仍可写入并回答示例问题。
    若 prompt 中包含默认文档的关键内容则返回对应关系，否则返回空。
    """
    # 与 DEFAULT_DOCUMENTS 内容对应：文档 1 收购、文档 2 任职、文档 3 产品与竞品
    if "BetaSolutions" in prompt and "收购" in prompt:
        return KnowledgeGraph(
            relationships=[
                Relationship(
                    source=Node(id="AlphaCorp", type="Company"),
                    target=Node(id="BetaSolutions", type="Company"),
                    type="ACQUIRED",
                )
            ]
        )
    if "Evelyn Reed" in prompt and "AlphaCorp" in prompt:
        return KnowledgeGraph(
            relationships=[
                Relationship(
                    source=Node(id="Evelyn Reed", type="Person"),
                    target=Node(id="AlphaCorp", type="Company"),
                    type="WORKS_FOR",
                )
            ]
        )
    if "NeuraGen" in prompt or "QuantumLeap" in prompt:
        return KnowledgeGraph(
            relationships=[
                Relationship(
                    source=Node(id="Innovate Inc.", type="Company"),
                    target=Node(id="NeuraGen", type="Product"),
                    type="PRODUCES",
                ),
                Relationship(
                    source=Node(id="AlphaCorp", type="Company"),
                    target=Node(id="QuantumLeap AI", type="Product"),
                    type="PRODUCES",
                ),
                Relationship(
                    source=Node(id="NeuraGen", type="Product"),
                    target=Node(id="QuantumLeap AI", type="Product"),
                    type="COMPETES_WITH",
                ),
                Relationship(
                    source=Node(id="Johnathan Miles", type="Person"),
                    target=Node(id="Innovate Inc.", type="Company"),
                    type="WORKS_FOR",
                ),
            ]
        )
    return KnowledgeGraph(relationships=[])


def _mock_structured(pyd_model: type[BaseModel], prompt: str) -> BaseModel:
    """未配置 API 时的占位结构化输出：KnowledgeGraph 使用演示 mock 填充内存图，其他模型返回空字典解析。"""
    if pyd_model == KnowledgeGraph:
        return _mock_knowledge_graph_for_demo(prompt)
    return pyd_model.model_validate({})


def init_llm() -> ModelScopeChat:
    """
    从环境变量读取 ModelScope 配置并构造 ModelScopeChat 实例。
    使用 temperature=0 以保证抽取与 Cypher 生成更稳定；extra_body 中开启 json_object 便于解析。
    """
    base_url = os.environ.get(
        "MODELSCOPE_BASE_URL", "https://api-inference.modelscope.cn/v1"
    )
    api_key = os.environ.get("MODELSCOPE_API_KEY", "")
    model_id = os.environ.get("MODELSCOPE_MODEL_ID", "deepseek-ai/DeepSeek-V3.2")
    extra = {
        "enable_thinking": True,
        "trust_request_chat_template": True,
        "response_format": {"type": "json_object"},
    }
    return ModelScopeChat(
        base_url=base_url,
        api_key=api_key,
        model=model_id,
        temperature=0,
        extra_body=extra,
    )


# =========================
# 3) 图存储：内存图（无 Neo4j 时使用）
# =========================
# 演示时若不配置 NEO4J_URI，则使用 InMemoryGraph：在内存中维护节点/边，并实现本脚本
# 所需的 add_graph_documents 与 query 接口；query 仅解析演示中的几种 Cypher 模式，
# 完整 Cypher 需使用 Neo4j。


class InMemoryGraph:
    """
    内存图：用字典和列表在进程内存储节点与边，接口与 LangChain Neo4jGraph 的
    add_graph_documents / schema / query 对齐，便于无 Neo4j 环境下跑通演示。
    """

    def __init__(self):
        self._nodes: Dict[str, str] = {}  # 实体 id -> 类型（如 Person, Company）
        self._edges: List[tuple] = []  # (源 id, 关系类型, 目标 id)

    @property
    def schema(self) -> str:
        """返回当前图的节点类型与关系类型描述，供「问题 → Cypher」提示中的 schema 占位使用。"""
        node_types = set(self._nodes.values())
        rel_types = set(e[1] for e in self._edges)
        return (
            f"Node properties: {[{'labels': [t], 'properties': [('id', 'STRING')]} for t in node_types]}\n"
            f"Relationships: {list(rel_types)}"
        )

    def add_graph_documents(
        self,
        graph_documents: List[Relationship],
        include_source: bool = False,
    ) -> None:
        """将一批关系（含内嵌节点）写入图：提取 source/target 节点与关系类型，加入 _nodes 与 _edges。"""
        for rel in graph_documents:
            src, tgt = rel.source, rel.target
            self._nodes[src.id] = src.type
            self._nodes[tgt.id] = tgt.type
            self._edges.append((src.id, rel.type, tgt.id))
        if DEBUG:
            console.print(f"[dim]已写入 {len(graph_documents)} 条关系[/dim]")

    def query(self, cypher: str) -> Any:
        """
        根据 Cypher 字符串做简单模式匹配并执行图遍历，返回与演示问题对应的结果列表。
        仅支持本脚本中用到的几类查询（如 WORKS_FOR、ACQUIRED、多跳竞品）；通用 Cypher 需 Neo4j。
        """
        cypher = re.sub(r"\s+", " ", cypher).strip()
        # 模式 1：查询某公司员工；支持 :Person/:Company 或 :Entity（schema 仅 Entity 时模型会生成 :Entity）
        m = re.search(
            r"MATCH\s+\((\w+):\w+\)\[:WORKS_FOR\]->.*?id:\s*['\"](\w+)['\"]\).*RETURN\s+(\w+)\.id",
            cypher,
            re.I,
        )
        if m:
            var, company_id, ret = m.group(1), m.group(2), m.group(3)
            for (s, t, o) in self._edges:
                if t == "WORKS_FOR" and o == company_id:
                    return [{"p.id": s}]
        # 模式 2：查询某公司收购了哪家公司，如 MATCH (...)-[:ACQUIRED]->(acquired_company:Company) RETURN acquired_company.id
        m = re.search(
            r"ACQUIRED\]->\s*\([^)]*\)[^R]*RETURN\s+(\w+)\.id",
            cypher,
            re.I,
        )
        if m:
            for (s, t, o) in self._edges:
                if t == "ACQUIRED":
                    return [{f"{m.group(1)}.id": o}]
        # 模式 3：多跳推理——收购 BetaSolutions 的公司 → 其产品 → 竞品 → 竞品公司
        if "ACQUIRED" in cypher and "COMPETES_WITH" in cypher:
            acquirer = None
            for (s, t, o) in self._edges:
                if t == "ACQUIRED" and o == "BetaSolutions":
                    acquirer = s
                    break
            if not acquirer:
                return []
            product = None
            for (s, t, o) in self._edges:
                if t == "PRODUCES" and s == acquirer:
                    product = o
                    break
            if not product:
                return []
            competitor_product = None
            for (s, t, o) in self._edges:
                if t == "COMPETES_WITH" and s == product:
                    competitor_product = o
                    break
            if not competitor_product:
                return []
            competitor_company = None
            for (s, t, o) in self._edges:
                if t == "PRODUCES" and o == competitor_product:
                    competitor_company = s
                    break
            if competitor_company:
                return [{"competitor_company.id": competitor_company}]
        return []


def get_graph():
    """
    获取图存储实例：若配置了 NEO4J_URI，则尝试连接 Neo4j 并清空图后返回 Neo4jGraph；
    否则或连接失败时返回 InMemoryGraph，保证脚本可直接运行。
    """
    uri = os.environ.get("NEO4J_URI")
    if uri:
        try:
            from langchain_community.graphs import Neo4jGraph
            g = Neo4jGraph(
                url=uri,
                username=os.environ.get("NEO4J_USERNAME", "neo4j"),
                password=os.environ.get("NEO4J_PASSWORD", ""),
            )
            g.query("MATCH (n) DETACH DELETE n")
            return g
        except Exception as e:
            console.print(f"[yellow]Neo4j 连接失败: {e}，使用内存图[/yellow]")
    return InMemoryGraph()


# =========================
# 4) 图构建智能体：文本 → 知识图谱 → 写入图
# =========================
# 图构建链：对一段非结构化文本调用 LLM，按 KnowledgeGraph 的 schema 抽取实体与关系；
# 调用方负责将返回的 relationships 写入图存储（add_graph_documents）。


def get_graph_maker_chain(llm: ModelScopeChat):
    """
    返回一个函数 invoke(text) -> KnowledgeGraph。
    内部使用 with_structured_output(KnowledgeGraph)，保证输出可解析为 Pydantic 模型；
    写入图由调用方根据返回的 relationships 调用 graph.add_graph_documents 完成。
    """
    extractor = llm.with_structured_output(KnowledgeGraph)

    def invoke(text: str) -> KnowledgeGraph:
        # 关系类型保持英文全大写（如 WORKS_FOR, ACQUIRED），与 Cypher/图查询约定一致
        prompt = (
            "你擅长从文本中抽取信息并构建知识图谱。请从下面文本中抽取所有实体（节点）和关系。\n"
            "要求：\n"
            "1. 关系类型用全大写英文动词，如 WORKS_FOR、ACQUIRED、PRODUCES、COMPETES_WITH。\n"
            "2. 实体类型用英文单数：Person（人）、Company（公司）、Product（产品）等，不要全部写成 Entity。\n\n"
            f"从以下文本中抽取知识图谱：\n\n{text}"
        )
        return extractor.invoke(prompt)

    return invoke


# =========================
# 5) 知识摄入：批量文档写入图
# =========================
# 演示用默认文档：三段企业相关叙述，分别包含收购、人物任职、产品与竞品关系，
# 用于构建一个小型「企业情报」知识图谱。实体名（AlphaCorp、BetaSolutions、Innovate Inc. 等）
# 保持与英文版一致，便于 InMemoryGraph.query 中的模式匹配与多跳演示。


DEFAULT_DOCUMENTS = [
    "2023 年 5 月 15 日，科技巨头 AlphaCorp 宣布收购云原生数据库技术领先的创业公司 BetaSolutions。",
    "AI 领域知名研究者 Evelyn Reed 博士自 2021 年起担任 AlphaCorp 首席科学官，负责 QuantumLeap AI 平台团队。",
    "Innovate Inc. 的主打产品 NeuraGen 被视为 AlphaCorp 的 QuantumLeap AI 的直接竞品；Innovate Inc. 近期聘请 Johnathan Miles 担任 CTO。",
]


def ingest_documents(
    graph,
    graph_maker_invoke,
    documents: List[str] = None,
) -> None:
    """逐段将文档交给图构建链抽取知识图谱，并把得到的关系列表写入图；打印每段结果及最终 schema。
    若抽取结果为空且为默认演示文档，则用预置 mock 数据填充，保证无 Neo4j/无 API 或 API 未返回关系时演示仍可跑通。
    """
    documents = documents or DEFAULT_DOCUMENTS
    for i, doc in enumerate(documents):
        console.print(f"--- 摄入文档 {i + 1} ---")
        try:
            kg = graph_maker_invoke(doc)
            if kg.relationships:
                graph.add_graph_documents(kg.relationships, include_source=False)
                console.print(
                    f"[green]已添加 {len(kg.relationships)} 条关系到图[/green]"
                )
            else:
                # 默认演示文档且未抽到关系时，用 mock 数据兜底（无 API 或 API 返回空时演示仍可回答示例问题）
                mock_prompt = "从以下文本中抽取知识图谱：\n\n" + doc
                kg_fallback = _mock_knowledge_graph_for_demo(mock_prompt)
                if kg_fallback.relationships:
                    graph.add_graph_documents(
                        kg_fallback.relationships, include_source=False
                    )
                    console.print(
                        f"[green]已用演示 mock 数据填充 {len(kg_fallback.relationships)} 条关系（API 未返回或未配置）[/green]"
                    )
                else:
                    console.print("[yellow]未抽取到关系[/yellow]")
        except Exception as e:
            console.print(f"[red]处理失败: {e}[/red]")
    console.print("--- [OK] 知识图谱摄入完成 ---")
    console.print("\n--- 图 Schema ---")
    console.print(graph.schema)


# =========================
# 6) 图查询智能体：问题 → Cypher → 执行 → 答案
# =========================
# Text-to-Cypher 流程：1）用 schema + 用户问题生成 Cypher（英文提示更利于模型输出合法 Cypher）；
# 2）在图存储上执行查询得到 context；3）用问题 + context 合成自然语言答案（与问题同语言）。


# Cypher 生成提示：保持英文，便于模型稳定输出符合 Neo4j 语法的查询，且不引入 schema 外的标签/关系类型
CYPHER_PROMPT = """You are an expert Neo4j Cypher developer. Convert the user's natural language question into a valid Cypher query.
Use ONLY the provided graph schema. Do not use node labels or relationship types not in the schema.
Return ONLY the Cypher query, no extra text.

Graph Schema:
{schema}

User Question:
{question}
"""

# 答案合成提示：与用户问题同语言回答（中文问题则用中文答）
ANSWER_PROMPT = """你是一个根据知识图谱查询结果回答用户问题的助手。
请根据下面的「查询结果」对「用户问题」给出简洁、准确的回答，使用与用户问题相同的语言（中文问题用中文答，英文问题用英文答）。

用户问题：{question}
查询结果：{context}
"""


def query_graph(
    graph,
    llm: ModelScopeChat,
    question: str,
) -> Dict[str, Any]:
    """
    完整的一轮图问答：先根据 schema 与问题生成 Cypher，再在图存储上执行，
    最后用问题与查询结果合成自然语言答案；返回 {"answer": str}。
    """
    console.print(f"\n[bold]问题：[/bold] {question}")
    console.print("--- >> 生成 Cypher 查询 ---")
    cypher_prompt = CYPHER_PROMPT.format(
        schema=graph.schema,
        question=question,
    )
    generated_cypher = llm.invoke(cypher_prompt, use_json_format=False)
    if DEBUG:
        console.print(f"[cyan]{generated_cypher}[/cyan]")
    console.print("--- 执行查询 ---")
    try:
        context = graph.query(generated_cypher)
        if isinstance(context, list):
            context = str(context)
        console.print(f"[yellow]查询结果：{context}[/yellow]")
    except Exception as e:
        console.print(f"[red]查询执行失败: {e}[/red]")
        return {"answer": "无法执行图查询以回答该问题。"}
    console.print("--- 合成最终答案 ---")
    answer_prompt = ANSWER_PROMPT.format(question=question, context=context)
    answer = llm.invoke(answer_prompt, use_json_format=False)
    return {"answer": answer}


# =========================
# 7) CLI 与入口
# =========================


def parse_args():
    """解析命令行：--question 指定单个问题，--ingest-only 只建图不问答，--debug 开启详细输出。"""
    import argparse
    p = argparse.ArgumentParser(
        description="图/世界模型记忆架构：知识图谱构建与多跳问答"
    )
    p.add_argument(
        "--question",
        type=str,
        default="",
        help="要回答的问题；不传则运行内置的三个演示问题（中文）",
    )
    p.add_argument(
        "--ingest-only",
        action="store_true",
        help="仅摄入文档构建图，不进行问答",
    )
    p.add_argument(
        "--debug",
        action="store_true",
        help="开启详细日志（如打印生成的 Cypher）",
    )
    return p.parse_args()


def main():
    """加载环境变量 → 初始化 LLM 与图 → 摄入默认文档 → 若无 --ingest-only 则依次回答默认或 --question 问题。"""
    global DEBUG
    load_dotenv()
    args = parse_args()
    DEBUG = getattr(args, "debug", False)

    if not os.environ.get("MODELSCOPE_API_KEY"):
        console.print(
            "[bold yellow]未设置 MODELSCOPE_API_KEY，将使用模拟响应[/bold yellow]"
        )

    llm = init_llm()
    graph = get_graph()
    graph_maker_invoke = get_graph_maker_chain(llm)

    ingest_documents(graph, graph_maker_invoke)

    if args.ingest_only:
        return

    questions = (
        [args.question]
        if args.question
        else [
            "谁在 AlphaCorp 工作？",
            "AlphaCorp 收购了哪家公司？",
            "收购 BetaSolutions 的那家公司所生产的产品，其竞品来自哪家公司？",
        ]
    )
    for q in questions:
        result = query_graph(graph, llm, q)
        console.print("\n--- 最终答案 ---")
        console.print(Markdown(result["answer"]))


if __name__ == "__main__":
    main()

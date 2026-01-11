#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
# 📘 智能体架构 8：情景记忆 + 语义记忆栈

欢迎来到我们系列的第八个文件。今天，我们将解决创建真正智能、长期助手的最关键挑战之一：**持久记忆**。标准聊天机器人的记忆是短暂的，只持续一个会话。要构建一个与用户一起学习和成长的个性化智能体，
我们需要更强大的解决方案。

我们将实现一个结构化的记忆架构，它模仿人类认知，结合了两种不同类型的记忆：

1.  **情景记忆**：这是对特定事件或过去交互的记忆。它回答"发生了什么？"的问题（例如，"上周，用户问我关于NVIDIA的股票价格。"）。我们将使用**向量数据库**来查找与当前主题相关的过去对话。
2.  **语义记忆**：这是从这些事件中提取的结构化事实、概念和关系的记忆。它回答"我知道什么？"的问题（例如，"用户Alex是保守投资者。"，"Alex对科技股感兴趣。"）。
    我们将使用**图数据库(Neo4j)**，因为它擅长管理和查询复杂关系。

通过结合这些，我们的智能体不仅可以回忆过去的对话，还可以建立一个关于用户和世界的丰富、相互关联的知识库，从而实现深度个性化和上下文感知的交互。

### 定义
**情景记忆 + 语义记忆栈**是一种智能体架构，它维护两种类型的长期记忆。
**情景记忆**存储经验的时间日志（例如，聊天历史摘要），通常基于语义相似性进行搜索。
**语义记忆**在知识库（通常是图graph）中存储从这些事件中提取的结构化知识（事实、实体、关系）。

### 高级工作流程

1.  **交互**：智能体与用户进行对话。
2.  **记忆检索（回忆）**：对于新的用户查询，智能体首先查询两个记忆系统。
    *   它在**情景**向量存储中搜索类似的过去对话。
    *   它在**语义**图数据库中查询与查询相关的实体和事实。
3.  **增强生成**：检索到的记忆被添加到提示的上下文中，允许LLM生成了解过去交互和学习事实的响应。
4.  **记忆创建（编码）**：交互完成后，后台进程分析对话。
    *   它创建该轮次的简明摘要（新的**情景**记忆）。
    *   它提取关键实体和关系（新的**语义**记忆）。
5.  **记忆存储**：新的情景摘要被嵌入并保存到向量存储中。新的语义事实作为节点和边写入图数据库。

### 何时使用 / 应用
*   **长期个人助手**：记住用户在数周或数月内的偏好、项目和个人详情的助手。
*   **个性化系统**：记住您风格的电子商务机器人，或记住您学习进度和弱点的教育导师。
*   **复杂研究智能体**：在探索文档时构建主题知识图的智能体，允许它回答复杂的多跳问题。

### 优势和劣势
*   **优势**：
    *   **真正的个性化**：实现了无限期持续的上下文和学习，远远超出单个会话的上下文窗口。
    *   **丰富的理解**：图数据库允许智能体理解和推理实体之间的复杂关系。
*   **劣势**：
    *   **复杂性**：这是一个比简单无状态智能体更复杂的架构来构建和维护。
    *   **记忆膨胀和修剪**：随着时间的推移，记忆存储会变得非常庞大。总结、巩固或修剪旧/不相关记忆的策略对于长期性能至关重要。
"""

# 📋 环境与设置
# 我们将安装所有必要的库，包括向量数据库驱动程序，并配置API密钥。
# 注意：使用Mock Neo4j，无需真实Neo4j数据库

# !pip install -q -U langchain-nebius langchain langgraph rich python-dotenv langchain_community langchain-openai neo4j faiss-cpu tiktoken

import os
import uuid
from typing import List, Dict, Any, Optional, Tuple
from dotenv import load_dotenv

# Pydantic用于数据建模
from pydantic import BaseModel, Field

# OpenAI相关
from openai import OpenAI
from openai import RateLimitError, APIError

# LangChain组件
from langchain_community.graphs import Neo4jGraph
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.embeddings import Embeddings


# 使用ModelScope嵌入模型的自定义嵌入类
class CustomEmbeddings(Embeddings):
    def __init__(self):
        # 从环境变量获取配置，使用嵌入模型专用的API密钥
        base_url = os.environ.get("MODELSCOPE_BASE_URL", "https://api-inference.modelscope.cn/v1")
        # 优先使用嵌入模型专用API密钥，如果不存在则使用通用密钥
        api_key = os.environ.get("MODELSCOPE_EMBEDDING_API_KEY", os.environ.get("MODELSCOPE_API_KEY", ""))
        
        # 创建OpenAI客户端指向ModelScope
        self.client = OpenAI(
            base_url=base_url,
            api_key=api_key
        )
        
        # 使用ModelScope的嵌入模型
        self.model = "Qwen/Qwen3-Embedding-0.6B"
    
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        # 批量获取嵌入
        response = self.client.embeddings.create(
            model=self.model,
            input=texts,
            encoding_format="float"
        )
        return [item.embedding for item in response.data]
    
    def embed_query(self, text: str) -> list[float]:
        # 获取单个文本的嵌入
        response = self.client.embeddings.create(
            model=self.model,
            input=[text],
            encoding_format="float"
        )
        return response.data[0].embedding

# LangGraph组件
from langgraph.graph import StateGraph, END
from typing_extensions import TypedDict

# 用于美化打印
from rich.console import Console
from rich.markdown import Markdown
import logging
from rich.logging import RichHandler

# 配置调试模式
DEBUG: bool = True

# 初始化日志
logger = logging.getLogger("memory_agent")
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
    return ModelScopeChat(base_url=base_url, api_key=api_key, model=model_id, temperature=0.2, extra_body=extra)

# --- API密钥和追踪设置 ---
load_dotenv()

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "Agentic Architecture - Memory Stack (ModelScope)"

# 检查所需的环境变量
required_vars = ["MODELSCOPE_API_KEY", "MODELSCOPE_EMBEDDING_API_KEY", "LANGCHAIN_API_KEY"]
missing_vars = []
for var in required_vars:
    if var not in os.environ:
        missing_vars.append(var)

if missing_vars:
    console.print(f"[yellow]警告: 以下环境变量未设置: {', '.join(missing_vars)}[/yellow]")
    console.print("[gray]注意: 使用Mock Neo4j，无需Neo4j环境变量[/gray]")

print("环境变量已加载，追踪已设置。")

# 🧠 构建记忆组件
# 这是我们架构的核心。我们将定义记忆的结构，建立与数据库的连接。我们还将创建"记忆创建者"智能体，负责处理对话并创建新记忆。

console = Console()

# 初始化LLM
llm = init_llm()

# 创建自定义嵌入（模拟版本）
embeddings = CustomEmbeddings()

# --- 1. 情景记忆的向量存储 ---
# 在实际应用中，您会持久化它。在本示例中，它在内存中。
try:
    episodic_vector_store = FAISS.from_texts(["Initial document to bootstrap the store"], embeddings)
except ImportError:
    console.print("[bold red]FAISS未安装。请运行`pip install faiss-cpu`。[/bold red]")
    episodic_vector_store = None

# --- 2. 语义记忆的图数据库 ---（含Mock实现）

# Mock Neo4j Graph 类 - 用于测试和演示，无需真实数据库连接
class MockNeo4jGraph:
    """模拟Neo4j图数据库的内存实现"""
    def __init__(self, **kwargs):
        # 内存存储
        self.nodes = {}
        self.relationships = []
        # 模拟索引
        self._fulltext_index = {}
        console.print("[yellow]使用Mock Neo4j Graph数据库（无需真实连接）[/yellow]")
    
    def query(self, cypher_query, params=None):
        """模拟Cypher查询执行"""
        params = params or {}
        console.print(f"[gray]执行Mock查询: {cypher_query[:50]}...[/gray]")
        
        query = cypher_query.strip().upper()
        
        # 处理清除图的查询
        if query.startswith("MATCH (N) DETACH DELETE N"):
            self.nodes = {}
            self.relationships = []
            self._fulltext_index = {}
            return "Graph cleared"
        
        # 处理全文检索查询
        if "DB.INDEX.FULLTEXT.QUERYNODES" in query and "ENTITY" in query:
            keywords = params.get("keywords", [])
            results = []
            
            for keyword in keywords:
                keyword_lower = keyword.lower()
                # 搜索包含关键词的节点
                for node_id, node in self.nodes.items():
                    # 检查节点ID、类型或属性是否包含关键词
                    if keyword_lower in node_id.lower() or \
                       keyword_lower in node["type"].lower() or \
                       any(keyword_lower in str(val).lower() for val in node["properties"].values()):
                        # 查找相关关系
                        for rel in self.relationships:
                            if rel["source"] == node_id or rel["target"] == node_id:
                                related_node_id = rel["target"] if rel["source"] == node_id else rel["source"]
                                related_node = self.nodes.get(related_node_id)
                                if related_node:
                                    results.append({
                                        "node": node,
                                        "relationship": rel,
                                        "related_node": related_node
                                    })
            
            # 限制结果数量
            if "LIMIT" in query:
                limit_str = query.split("LIMIT")[-1].strip()
                limit = int(limit_str.split()[0]) if limit_str else 5
                results = results[:limit]
            
            return results
        
        # 处理关系查询 - MATCH (n)-[r]->(m)
        if "MATCH" in query and "-[R" in query and "->" in query:
            # 解析关系类型
            rel_types = []
            if "INTERESTED_IN" in query:
                rel_types.append("INTERESTED_IN")
            if "HAS_GOAL" in query:
                rel_types.append("HAS_GOAL")
            if "IS_A" in query:
                rel_types.append("IS_A")
            
            results = []
            for rel in self.relationships:
                # 如果没有指定关系类型或关系类型匹配
                if not rel_types or rel["type"] in rel_types:
                    results.append({
                        "n": self.nodes.get(rel["source"]),
                        "r": rel,
                        "m": self.nodes.get(rel["target"])
                    })
            return results
        
        # 处理节点查询 - MATCH (n:Label)
        if "MATCH (N:" in query:
            # 简单实现，返回所有节点
            return list(self.nodes.values())
        
        return []
    
    def add_graph_documents(self, documents, include_source=False):
        """模拟添加图文档"""
        added_relationships = 0
        
        for doc in documents:
            if hasattr(doc, "source") and hasattr(doc, "target") and hasattr(doc, "type"):
                # 处理Relationship对象
                source_node = doc.source
                target_node = doc.target
                relationship_type = doc.type
                
                # 确保节点ID唯一
                source_id = source_node.id
                target_id = target_node.id
                
                # 添加源节点
                if source_id not in self.nodes:
                    self.nodes[source_id] = {
                        "id": source_id,
                        "type": source_node.type,
                        "properties": source_node.properties.copy()
                    }
                    # 更新全文索引
                    self._update_fulltext_index(source_node)
                else:
                    # 更新现有节点的属性
                    existing_node = self.nodes[source_id]
                    existing_node["properties"].update(source_node.properties)
                    self._update_fulltext_index(source_node)
                
                # 添加目标节点
                if target_id not in self.nodes:
                    self.nodes[target_id] = {
                        "id": target_id,
                        "type": target_node.type,
                        "properties": target_node.properties.copy()
                    }
                    # 更新全文索引
                    self._update_fulltext_index(target_node)
                else:
                    # 更新现有节点的属性
                    existing_node = self.nodes[target_id]
                    existing_node["properties"].update(target_node.properties)
                    self._update_fulltext_index(target_node)
                
                # 创建关系
                relationship = {
                    "source": source_id,
                    "target": target_id,
                    "type": relationship_type,
                    "properties": doc.properties.copy()
                }
                
                # 检查关系是否已存在
                rel_exists = False
                for existing_rel in self.relationships:
                    if (existing_rel["source"] == source_id and 
                        existing_rel["target"] == target_id and 
                        existing_rel["type"] == relationship_type):
                        # 更新现有关系的属性
                        existing_rel["properties"].update(doc.properties)
                        rel_exists = True
                        break
                
                if not rel_exists:
                    self.relationships.append(relationship)
                    added_relationships += 1
        
        return added_relationships
    
    def _update_fulltext_index(self, node):
        """更新全文索引"""
        # 更完善的全文索引实现
        texts_to_index = [
            node.id.lower(),
            node.type.lower()
        ]
        
        # 添加属性值到索引
        for val in node.properties.values():
            if val is not None:
                texts_to_index.append(str(val).lower())
        
        # 创建所有文本的组合
        all_text = " ".join(texts_to_index)
        
        # 分割成单词并添加到索引
        for word in all_text.split():
            # 过滤掉过短的单词
            if len(word) > 1:
                if word not in self._fulltext_index:
                    self._fulltext_index[word] = set()
                self._fulltext_index[word].add(node.id)
    
    def get_node_count(self):
        """获取节点数量"""
        return len(self.nodes)
    
    def get_relationship_count(self):
        """获取关系数量"""
        return len(self.relationships)
    
    def clear(self):
        """清除所有数据"""
        self.nodes = {}
        self.relationships = []
        self._fulltext_index = {}
    
    @property
    def get_schema(self):
        """模拟获取图结构（作为属性）"""
        return {
            "nodes": [{
                "labels": list(set(node["type"] for node in self.nodes.values())),
                "properties": {}
            }],
            "relationships": [{
                "type": rel["type"],
                "properties": {}
            } for rel in self.relationships]
        }

# 使用Mock实现替代真实Neo4j连接
try:
    graph = MockNeo4jGraph()
    # 清除图以进行干净运行
    graph.query("MATCH (n) DETACH DELETE n")
except Exception as e:
    console.print(f"[bold red]初始化Mock图数据库失败: {e}[/bold red]")
    graph = None

# --- 3. "记忆创建者"的Pydantic模型 ---
# 定义我们要提取的知识结构。
class Node(BaseModel):
    id: str = Field(description="节点的唯一标识符，可以是人名、公司股票代码或概念。")
    type: str = Field(description="节点的类型（例如，'User'、'Company'、'InvestmentPhilosophy'）。")
    properties: Dict[str, Any] = Field(description="节点的属性字典。")

class Relationship(BaseModel):
    source: Node = Field(description="关系的源节点。")
    target: Node = Field(description="关系的目标节点。")
    type: str = Field(description="关系的类型（例如，'IS_A'、'INTERESTED_IN'）。")
    properties: Dict[str, Any] = Field(description="关系的属性字典。")

class KnowledgeGraph(BaseModel):
    """表示从对话中提取的结构化知识。"""
    relationships: List[Relationship] = Field(description="要添加到知识图的关系列表。")

# --- 4. "记忆创建者"智能体 ---
def create_memories(user_input: str, assistant_output: str):
    """创建情景记忆和语义记忆"""
    conversation = f"User: {user_input}\nAssistant: {assistant_output}"
    
    try:
        # 4a. 创建情景记忆（摘要）
        console.print("--- 创建情景记忆（摘要） ---")
        summary_prompt = ChatPromptTemplate.from_messages([
            ("system", "您是总结专家。为以下用户-助手交互创建简洁的一句话总结。该总结将用作未来回忆的记忆。"),
            ("human", "交互:\n{interaction}")
        ])
        # 格式化提示并调用LLM
        formatted_prompt = summary_prompt.format_messages(interaction=conversation)
        formatted_prompt_str = formatted_prompt[0].content + "\n\n" + formatted_prompt[1].content
        episodic_summary = llm.invoke(formatted_prompt_str)
        
        if episodic_vector_store is not None:
            new_doc = Document(page_content=episodic_summary, metadata={"created_at": uuid.uuid4().hex, "source": "conversation"})
            episodic_vector_store.add_documents([new_doc])
            console.print(f"[green]情景记忆已创建:[/green] '{episodic_summary}'")
        else:
            console.print("[red]无法创建情景记忆: 向量存储未初始化[/red]")
        
        # 4b. 创建语义记忆（事实提取）
        console.print("--- 创建语义记忆（图） ---")
        extraction_llm = llm.with_structured_output(KnowledgeGraph)
        extraction_prompt = ChatPromptTemplate.from_messages([
            ("system", "您是知识提取专家。您的任务是从对话中识别关键实体及其关系，并将它们建模为图。关注用户偏好、目标和陈述的事实。"),
            ("human", "从这个交互中提取所有关系:\n{interaction}")
        ])
        # 格式化提示并调用LLM
        formatted_prompt = extraction_prompt.format_messages(interaction=conversation)
        formatted_prompt_str = formatted_prompt[0].content + "\n\n" + formatted_prompt[1].content
        
        kg_data = extraction_llm.invoke(formatted_prompt_str)
        
        if graph is not None:
            if kg_data.relationships:
                added = 0
                for rel in kg_data.relationships:
                    added += graph.add_graph_documents([rel], include_source=True)
                if added > 0:
                    console.print(f"[green]语义记忆已创建:[/green] 添加了 {added} 个新关系到图中。")
                else:
                    console.print("[yellow]未添加新的语义记忆关系（所有关系已存在）[/yellow]")
            else:
                console.print("[yellow]在此交互中未识别到新的语义记忆。[/yellow]")
        else:
            console.print("[red]无法创建语义记忆: 图数据库未初始化[/red]")
    
    except Exception as e:
        console.print(f"[bold red]创建记忆时出错: {str(e)}[/bold red]")
        logger.error(f"创建记忆失败: {str(e)}")

if episodic_vector_store and graph:
    print("记忆组件初始化成功。")

# 🤖 记忆增强的智能体
# 现在我们将构建使用此记忆系统的智能体。我们将使用LangGraph来定义一个清晰的、有状态的工作流：检索记忆，使用这些记忆生成响应，最后，用最新的交互更新记忆。

# 定义我们LangGraph智能体的状态
class AgentState(TypedDict):
    user_input: str
    retrieved_memories: Optional[str]
    generation: str

# 定义图的节点
def retrieve_memory(state: AgentState) -> Dict[str, Any]:
    """从情景和语义存储中检索记忆的节点。"""
    console.print("--- 检索记忆 ---")
    user_input = state['user_input']
    
    # 从情景记忆中检索
    episodic_memories = ""
    if episodic_vector_store is not None:
        try:
            retrieved_docs = episodic_vector_store.similarity_search(user_input, k=2)
            episodic_memories = "\n".join([doc.page_content for doc in retrieved_docs])
        except Exception as e:
            console.print(f"[red]检索情景记忆时出错: {e}[/red]")
            episodic_memories = "无法检索情景记忆"
    else:
        episodic_memories = "情景记忆存储未初始化"
    
    # 从语义记忆中检索
    semantic_memories = ""
    if graph is not None:
        try:
            graph_schema = graph.get_schema
            # 使用全文索引以获得更好的检索效果
            semantic_memories = str(graph.query("""
                UNWIND $keywords AS keyword
                CALL db.index.fulltext.queryNodes("entity", keyword) YIELD node, score
                MATCH (node)-[r]-(related_node)
                RETURN node, r, related_node LIMIT 5
                """, {'keywords': user_input.split()}))
        except Exception as e:
            semantic_memories = f"无法查询图: {e}"
    else:
        semantic_memories = "语义记忆存储未初始化"
        
    retrieved_content = f"相关过去对话（情景记忆）:\n{episodic_memories}\n\n相关事实（语义记忆）:\n{semantic_memories}"
    console.print(f"[cyan]检索到的上下文:\n{retrieved_content}[/cyan]")
    
    return {"retrieved_memories": retrieved_content}

def generate_response(state: AgentState) -> Dict[str, Any]:
    """使用检索到的记忆生成响应的节点。"""
    console.print("--- 生成响应 ---")
    prompt = ChatPromptTemplate.from_messages([
        ("system", "您是一位有帮助且个性化的财务助手。使用检索到的记忆来提供信息并根据用户定制您的响应。如果记忆表明用户的偏好（例如，他们是保守投资者），您必须尊重这一点。"),
        ("human", "我的问题是: {user_input}\n\n以下是一些可能相关的记忆:\n{retrieved_memories}")
    ])
    # 格式化提示并调用LLM
    formatted_prompt = prompt.format_messages(user_input=state['user_input'], retrieved_memories=state['retrieved_memories'])
    formatted_prompt_str = formatted_prompt[0].content + "\n\n" + formatted_prompt[1].content
    generation = llm.invoke(formatted_prompt_str)
    console.print(f"[green]生成的响应:\n{generation}[/green]")
    return {"generation": generation}

def update_memory(state: AgentState) -> Dict[str, Any]:
    """使用最新交互更新记忆的节点。"""
    console.print("--- 更新记忆 ---")
    create_memories(state['user_input'], state['generation'])
    return {}

# 构建图
workflow = StateGraph(AgentState)

workflow.add_node("retrieve", retrieve_memory)
workflow.add_node("generate", generate_response)
workflow.add_node("update", update_memory)

workflow.set_entry_point("retrieve")
workflow.add_edge("retrieve", "generate")
workflow.add_edge("generate", "update")
workflow.add_edge("update", END)

memory_agent = workflow.compile()
print("记忆增强的智能体图编译成功。")

# 🚀 演示与检查
# 让我们看看智能体的实际应用。我们将模拟一个多轮对话。前两轮将为记忆提供种子。第三轮将测试智能体是否可以使用该记忆来提供个性化响应。最后，我们将直接检查数据库，查看创建的记忆。

def run_interaction(query: str):
    result = memory_agent.invoke({"user_input": query})
    return result['generation']

console.print("\n--- 💬 交互 1: 为记忆提供种子 ---")
run_interaction("你好，我叫Alex。我是一个保守的投资者，主要对成熟的科技公司感兴趣。")

console.print("\n--- 💬 交互 2: 询问具体问题 ---")
run_interaction("你对苹果(AAPL)有什么看法？")

console.print("\n--- 🧠 交互 3: 记忆测试 ---")
run_interaction("根据我的目标，有什么好的替代股票？")

# 🔍 检查记忆存储
# 让我们看看引擎盖下的情况。我们可以直接查询我们的数据库，查看智能体创建的记忆。

console.print("--- 🔍 检查情景记忆（向量存储） ---")
if episodic_vector_store is not None:
    try:
        # 我们将对一般概念进行相似性搜索，看看会出现什么
        retrieved_docs = episodic_vector_store.similarity_search("用户的投资策略", k=3)
        if retrieved_docs:
            for i, doc in enumerate(retrieved_docs):
                print(f"{i+1}. {doc.page_content}")
        else:
            print("[yellow]未找到相关情景记忆[/yellow]")
    except Exception as e:
        print(f"[red]查询情景记忆时出错: {e}[/red]")
else:
    print("[red]情景记忆存储未初始化[/red]")

console.print("\n--- 🕸️ 检查语义记忆（图数据库） ---")
if graph is not None:
    try:
        print(f"图模式:\n{graph.get_schema}")
        print(f"节点数量: {graph.get_node_count()}")
        print(f"关系数量: {graph.get_relationship_count()}")
        
        # Cypher查询，查看谁对什么感兴趣
        query_result = graph.query("MATCH (n:User)-[r:INTERESTED_IN|HAS_GOAL|IS_A]->(m) RETURN n, r, m")
        if query_result:
            print(f"图中的关系:\n{query_result}")
        else:
            print("[yellow]图中未找到关系[/yellow]")
    except Exception as e:
        print(f"[red]查询图数据库时出错: {e}[/red]")
else:
    print("[red]图数据库未初始化[/red]")

# 📝 结论
# 在这个文件中，我们成功构建了一个具有复杂长期记忆系统的智能体。演示清楚地展示了这种架构的强大功能：
# 
# - **无状态失败**：标准智能体在被问及"根据我的目标，有什么好的替代方案？"时会失败，因为它没有用户目标的记忆。
# - **记忆增强成功**：我们的智能体成功了，因为它可以：
#     1.  **情景回忆**：它检索了第一次对话的摘要："用户Alex介绍自己是保守投资者..."
#     2.  **语义回忆**：它查询了图并找到了结构化事实：`(User: Alex) -[HAS_GOAL]-> (InvestmentPhilosophy: Conservative)`。
#     3.  **综合**：它使用这个组合上下文提供了高度相关和个性化的建议（微软），明确引用了用户的保守目标。
# 
# 这种回忆"发生了什么"（情景）和"知道什么"（语义）的结合是一个强大的范例，它超越了简单的交易型智能体，创造了真正的学习伙伴。虽然大规模管理这种记忆会带来修剪和整合等挑战，但我们在这里构建的基础架构是朝着更智能、更个性化的AI系统迈出的重要一步。
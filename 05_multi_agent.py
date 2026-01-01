# -*- coding: utf-8 -*-
"""
多智能体系统（Multi-Agent System）的可运行示例

学习目标（看完你能做到什么）：
- 理解什么是多智能体系统：由多个专业化智能体组成的团队协作解决复杂问题
- 掌握如何使用LangGraph1.0构建多智能体工作流
- 学会设计不同角色的智能体（技术分析师、新闻分析师、财务分析师、经理）
- 理解单智能体与多智能体系统的区别和各自的优势

核心概念速览：
- 角色专业化：每个智能体有明确的角色、技能和职责
- 任务分解：复杂问题被分解为子任务，分配给最适合的智能体
- 协作流程：智能体通过工作流进行通信和协作
- 结果合成：经理智能体将各专家的输出整合成最终报告

运行前准备：
- 项目根目录创建 `.env` 并配置：
  - `MODELSCOPE_API_KEY`（必需，ModelScope 推理令牌）
  - `MODELSCOPE_BASE_URL`（可选，默认：https://api-inference.modelscope.cn/v1）
  - `MODELSCOPE_MODEL_ID`（可选，默认：deepseek-ai/DeepSeek-V3.2）
  - `LANGCHAIN_API_KEY`（可选，用于 LangSmith 追踪）

如何运行：
- 直接运行默认示例：`python 05_multi_agent.py`
- 自定义公司：`python 05_multi_agent.py --company "阿里巴巴 (BABA)"`

阅读建议：
- 先从"数据结构与模型定义"开始理解
- 然后看"LLM初始化"部分了解如何构建智能体使用的语言模型
- 接着学习"智能体节点"实现，理解每个角色的职责
- 最后看"工作流构建"了解如何将各个智能体组织起来协作
"""

import os
import re
import json
import argparse
from typing import List, Annotated, TypedDict, Optional

from dotenv import load_dotenv

from pydantic import BaseModel, Field

# LangChain components
from langchain_core.messages import BaseMessage, ToolMessage, SystemMessage, HumanMessage, AIMessage
from langchain_core.tools import tool
from langchain_tavily import TavilySearch

# LangGraph components
from langgraph.graph import StateGraph, END
from langgraph.graph.message import AnyMessage, add_messages
from langgraph.prebuilt import ToolNode, tools_condition

# For pretty printing
from rich.console import Console
from rich.markdown import Markdown
from rich.logging import RichHandler

import logging
from openai import OpenAI, RateLimitError, APIError

# 全局日志收集器
GLOBAL_LOGS = []

# =========================
# 1) 数据结构与模型定义
# =========================
class AgentState(TypedDict):
    """智能体共享状态：在节点间传递的消息列表"""
    messages: Annotated[list[AnyMessage], add_messages]

# =========================
# 2) LLM 初始化与工具定义
# =========================
class ModelScopeChat:
    """
    ModelScope 的 OpenAI 兼容接口适配器：
    - 提供 invoke(prompt) 基本调用
    - 提供 with_structured_output(PydanticModel) 的结构化输出包装
    
    初学者理解要点：
    - 为什么需要“适配器”？因为我们的工作流依赖“结构化输出”，而不少服务默认只返回纯文本。
    - 适配器会尽量要求模型“只输出 JSON”，再解析为 Pydantic v2 模型；这样后续节点就能稳稳地拿到字段，而不是杂乱的文本。
    """
    def __init__(self, base_url: str, api_key: str, model: str, fallback_model: Optional[str] = None, temperature: float = 0.2, extra_body: Optional[dict] = None):
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.model = model
        self.fallback_model = fallback_model
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
                console.print(f"[bold yellow]⚠️ 主模型请求失败：{e}，尝试切换到备用模型[/bold yellow]")
                self.model = self.fallback_model
                self.switched = True
                resp = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=self.temperature,
                    stream=False,
                    extra_body=self.extra_body,
                )
                # 非流式返回：choices[0].message.content
                return resp.choices[0].message.content
            else:
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
                    content = resp.choices[0].message.content or ""
                except (RateLimitError, APIError) as e:
                    if not self.outer.switched and self.outer.fallback_model:
                        console.print(f"[bold yellow]⚠️ 主模型请求失败：{e}，尝试切换到备用模型[/bold yellow]")
                        self.outer.model = self.outer.fallback_model
                        self.outer.switched = True
                        resp = self.outer.client.chat.completions.create(
                            model=self.outer.model,
                            messages=messages,
                            temperature=self.outer.temperature,
                            stream=False,
                            extra_body=self.outer.extra_body,
                        )
                        content = resp.choices[0].message.content or ""
                    else:
                        raise
                def _extract_json(s: str) -> str:
                    m = re.search(r'(\{[\s\S]*\}|\[[\s\S]*\])', s)
                    return m.group(1) if m else "{}"
                raw = content.strip()
                try:
                    data = json.loads(raw)
                except Exception:
                    data = json.loads(_extract_json(raw))
                # 兜底字段映射：尽量把常见别名映射到目标模型字段
                try:
                    parsed = pyd_model.model_validate(data)
                    return parsed
                except Exception:
                    # DraftCode: 常见返回 'function' 字段，映射到 'code'
                    if "code" not in data and "function" in data:
                        data["code"] = data.pop("function")
                    if "explanation" not in data and "desc" in data:
                        data["explanation"] = data.pop("desc")
                    # RefinedCode: 将 'code' 映射为 'refined_code'
                    if "refined_code" not in data and "code" in data:
                        data["refined_code"] = data.pop("code")
                    # Critique: 可能给 'summary' 映射为 'critique_summary'
                    if "critique_summary" not in data and "summary" in data:
                        data["critique_summary"] = data.pop("summary")
                    parsed = pyd_model.model_validate(data)
                    return parsed

        return _StructuredWrapper(self)

def init_llm() -> ModelScopeChat:
    """
    初始化 ModelScope LLM（OpenAI 兼容接口）。
    - 可通过环境变量控制：
      MODELSCOPE_BASE_URL（默认：https://api-inference.modelscope.cn/v1）
      MODELSCOPE_API_KEY
      MODELSCOPE_MODEL_ID（默认：deepseek-ai/DeepSeek-V3.2）
      MODELSCOPE_MODEL_ID_R1（备用模型，可选）
    - 当主模型请求失败时，会自动切换到备用模型（如果配置了的话）
    """
    # 从环境变量获取配置，如果没有则使用默认值
    base_url = os.environ.get("MODELSCOPE_BASE_URL", "https://api-inference.modelscope.cn/v1")
    api_key = os.environ.get("MODELSCOPE_API_KEY", "")  # API密钥通常需要用户提供
    model_id = os.environ.get("MODELSCOPE_MODEL_ID", "deepseek-ai/DeepSeek-V3.2")  # 默认使用DeepSeek-V3.2模型
    fallback_model_id = os.environ.get("MODELSCOPE_MODEL_ID_R1")  # 备用模型
    
    # 额外参数设置
    extra = {
        "enable_thinking": True,  # 启用思考过程输出
        "trust_request_chat_template": True,  # 信任请求中的聊天模板
        "response_format": {"type": "json_object"}  # 要求模型返回JSON格式
    }
    
    # 创建并返回ModelScopeChat实例
    return ModelScopeChat(base_url=base_url, api_key=api_key, model=model_id, fallback_model=fallback_model_id, temperature=0.2, extra_body=extra)

# 创建一个包装器，使 ModelScopeChat 支持工具调用
class ModelScopeChatWithTools:
    """
    ModelScopeChat 的工具调用包装器：
    - 支持将工具绑定到 LLM
    - 处理工具调用请求和响应
    """
    def __init__(self, llm_instance: ModelScopeChat, tools: list):
        self.llm = llm_instance
        self.tools = tools
    
    def invoke(self, messages: list):
        # 将消息转换为提示字符串
        prompt = ""
        for msg in messages:
            if isinstance(msg, SystemMessage):
                prompt += f"[系统] {msg.content}\n"
            elif isinstance(msg, HumanMessage):
                prompt += f"[用户] {msg.content}\n"
            elif isinstance(msg, AIMessage):
                prompt += f"[助手] {msg.content}\n"
            else:
                # 处理其他消息类型
                try:
                    prompt += f"[消息] {msg.content}\n"
                except:
                    continue
        
        # 生成响应
        response = self.llm.invoke(prompt)
        
        # 解析工具调用
        tool_calls = []
        if "web_search" in response and "query" in response:
            try:
                tool_data = json.loads(response)
                if "tool_call" in tool_data:
                    tool_calls = [{
                        "name": tool_data["tool_call"]["name"],
                        "args": tool_data["tool_call"]["args"]
                    }]
            except:
                # 如果解析失败，检查简单模式
                match = re.search(r'web_search\(query=[\'"]([^\'"]+)[\'"]\)', response)
                if match:
                    tool_calls = [{
                        "name": "web_search",
                        "args": {"query": match.group(1)}
                    }]
        
        # 创建并返回实际的 AIMessage 对象
        return AIMessage(content=response, tool_calls=tool_calls)

# 简单的模拟网络搜索工具用于测试
@tool
def web_search(query: str) -> str:
    """模拟网络搜索并返回模拟结果。
    
    为了避免依赖外部API，我们使用预定义的模拟数据来模拟网络搜索结果。
    当查询包含特定关键词时，返回对应的模拟数据；否则返回查询内容的简单提示。
    """
    console = Console()
    console.print(f"--- 工具：搜索 '{query}'...")
    
    # 预定义的模拟数据，涵盖了常见的金融和技术查询
    mock_data = {
        "NVIDIA 最近新闻": "NVIDIA 发布了新一代 AI 芯片 H200，性能比上一代提升 2.5 倍。同时，NVIDIA 与多家云服务提供商达成合作，扩大其 AI 基础设施业务。",
        "NVIDIA 股价趋势": "NVIDIA 股价在过去 6 个月上涨了 85%，主要受 AI 芯片需求激增和良好的财报表现推动。技术指标显示，该股目前处于强势上涨趋势中。",
        "NVIDIA 财务表现": "NVIDIA 2023 财年第四季度营收达到 221 亿美元，同比增长 265%；净利润为 123 亿美元，同比增长 769%。数据中心业务收入占总营收的 70% 以上。",
        "NVIDIA 市场情绪": "分析师普遍看好 NVIDIA 的前景，有 45 位分析师给予买入评级，平均目标价为 1,000 美元。投资者对公司的 AI 芯片业务充满信心。",
        "NVIDIA 技术分析": "从技术面来看，NVIDIA 股价突破了多个关键阻力位，成交量放大，MACD 指标显示多头动能强劲。短期支撑位在 850 美元，阻力位在 950 美元。",
        "NVIDIA 财务分析": "NVIDIA 的毛利率达到 76.0%，净利率为 55.8%，均创历史新高。公司的自由现金流为 140 亿美元，财务状况非常健康。",
        # 添加更多公司的模拟数据
        "阿里巴巴 最近新闻": "阿里巴巴宣布进行组织架构调整，成立六大业务集团，各集团将独立融资和上市。同时，阿里巴巴加大对AI技术的投入，推出通义千问大模型。",
        "阿里巴巴 股价趋势": "阿里巴巴股价在过去 6 个月下跌了 15%，主要受宏观经济环境和监管政策影响。技术指标显示，该股目前处于盘整阶段。",
        "阿里巴巴 财务表现": "阿里巴巴2024财年第二季度营收达2247亿元人民币，同比增长9%；净利润达488亿元人民币，同比增长19%。云计算业务收入达276亿元人民币，同比增长2%。",
    }
    
    # 检查查询是否包含预定义的关键词（不区分大小写）
    for key, value in mock_data.items():
        if key in query:
            return value
    
    # 如果没有匹配的关键词，返回默认结果
    return f"模拟搜索结果：{query}"

# =========================
# 3) 智能体节点实现
# =========================
def build_monolithic_agent(llm_with_tools):
    """
    构建单智能体系统（作为基线对比）
    
    参数：
        llm_with_tools: 带工具的 LLM 实例
    
    返回：
        编译后的 LangGraph 应用
    """
    # 创建控制台实例用于打印信息
    console = Console()
    
    # ========================================================
    # 1. 定义单智能体节点：处理所有任务的单一智能体
    # ========================================================
    def monolithic_agent_node(state: AgentState):
        # 打印当前智能体的状态
        console.print("--- 单智能体：思考中... ---")
        
        # 调用带工具的LLM处理当前状态的消息
        # 输入：当前状态中的所有消息（包括用户查询和历史交互）
        # 输出：LLM的响应（可能包含工具调用请求）
        response = llm_with_tools.invoke(state["messages"])
        
        # 返回更新后的状态：将新的响应消息追加到消息列表中
        return {"messages": [response]}
    
    # ========================================================
    # 2. 创建工具节点：自动执行智能体请求的工具调用
    # ========================================================
    # ToolNode是LangGraph提供的预构建节点，能够自动：
    # - 检测AIMessage中的tool_calls字段
    # - 执行对应的工具
    # - 返回包含工具执行结果的ToolMessage
    tool_node = ToolNode([web_search])  # 使用模拟的web_search工具
    
    # ========================================================
    # 3. 构建ReAct（Reasoning + Acting）工作流
    # ========================================================
    # 创建状态图构建器，指定状态类型为AgentState
    mono_graph_builder = StateGraph(AgentState)
    
    # 添加节点到图中
    # 节点1："agent" - 单智能体节点，负责思考和决策
    mono_graph_builder.add_node("agent", monolithic_agent_node)
    # 节点2："tools" - 工具节点，负责执行工具调用
    mono_graph_builder.add_node("tools", tool_node)
    
    # 设置工作流的入口点：从agent节点开始执行
    mono_graph_builder.set_entry_point("agent")
    
    # ========================================================
    # 4. 定义条件边：根据agent节点的输出决定下一个节点
    # ========================================================
    def tools_condition_with_end(state):
        # 使用LangGraph预定义的tools_condition函数检查是否需要调用工具
        # tools_condition会检查最后一个AIMessage是否包含tool_calls
        result = tools_condition(state)
        
        if isinstance(result, str):
            # 旧版本的tools_condition返回字符串
            # 如果需要调用工具：返回 "tools"
            # 如果不需要调用工具：返回 "agent"（继续思考）
            return {result: "tools", "__default__": END}
        elif isinstance(result, dict):
            # 新版本的tools_condition返回映射字典
            # 键：工具调用的输出标签
            # 值：下一个要执行的节点
            result["__default__"] = END  # 添加默认路径：结束工作流
            return result
        else:
            # 处理意外情况：tools_condition返回了非预期的类型
            raise TypeError(f"tools_condition 返回意外类型: {type(result)}")
    
    # 添加条件边：从agent节点到tools_condition_with_end
    # 这意味着agent节点执行后，会根据tools_condition_with_end的返回值决定下一个节点
    mono_graph_builder.add_conditional_edges("agent", tools_condition_with_end)
    
    # 添加普通边：从tools节点回到agent节点
    # 这意味着工具执行完成后，会回到agent节点继续思考
    mono_graph_builder.add_edge("tools", "agent")
    
    # ========================================================
    # 5. 编译工作流并返回
    # ========================================================
    # 编译后的工作流是一个可调用的对象，可以接收初始状态并执行
    return mono_graph_builder.compile()

# 多智能体系统的各个节点

def news_analyst_node(state: AgentState):
    """
    新闻分析师智能体：分析公司的最新新闻和市场情绪
    
    参数：
        state: 智能体状态，包含消息历史
    
    返回：
        更新后的状态，包含新闻分析结果
    """
    log_message = "--- 新闻分析师：分析最新新闻和市场情绪... ---"
    GLOBAL_LOGS.append(log_message)
    console = Console()
    console.print(log_message)
    
    # 提取公司名称
    company = extract_company_name(state["messages"])
    
    # 构建新闻分析师的系统提示
    news_analyst_prompt = [
        SystemMessage(content=f"你是一位专业的新闻分析师，专注于分析{company}的最新新闻和市场情绪。"),
        SystemMessage(content="请总结最近的重要新闻，分析市场对该公司的整体情绪，并提供相关见解。"),
        SystemMessage(content="你可以使用 web_search 工具查找最新信息。")
    ] + state["messages"]
    
    # 生成响应
    llm = init_llm()
    llm_with_tools = ModelScopeChatWithTools(llm, [web_search])
    response = llm_with_tools.invoke(news_analyst_prompt)
    
    return {"messages": [AIMessage(content=f"## 新闻与市场情绪分析\n\n{response.content}")]}

def technical_analyst_node(state: AgentState):
    """
    技术分析师智能体：分析公司股票的技术走势
    
    参数：
        state: 智能体状态，包含消息历史
    
    返回：
        更新后的状态，包含技术分析结果
    """
    log_message = "--- 技术分析师：分析股票技术走势... ---"
    GLOBAL_LOGS.append(log_message)
    console = Console()
    console.print(log_message)
    
    # 提取公司名称
    company = extract_company_name(state["messages"])
    
    # 构建技术分析师的系统提示
    technical_analyst_prompt = [
        SystemMessage(content=f"你是一位专业的技术分析师，专注于分析{company}的股票技术走势。"),
        SystemMessage(content="请分析股票的价格趋势、关键支撑位和阻力位、技术指标以及潜在的交易信号。"),
        SystemMessage(content="你可以使用 web_search 工具查找最新的价格数据和技术指标。")
    ] + state["messages"]
    
    # 生成响应
    llm = init_llm()
    llm_with_tools = ModelScopeChatWithTools(llm, [web_search])
    response = llm_with_tools.invoke(technical_analyst_prompt)
    
    return {"messages": [AIMessage(content=f"## 技术分析\n\n{response.content}")]}

def financial_analyst_node(state: AgentState):
    """
    财务分析师智能体：分析公司的财务表现
    
    参数：
        state: 智能体状态，包含消息历史
    
    返回：
        更新后的状态，包含财务分析结果
    """
    log_message = "--- 财务分析师：分析公司财务表现... ---"
    GLOBAL_LOGS.append(log_message)
    console = Console()
    console.print(log_message)
    
    # 提取公司名称
    company = extract_company_name(state["messages"])
    
    # 构建财务分析师的系统提示
    financial_analyst_prompt = [
        SystemMessage(content=f"你是一位专业的财务分析师，专注于分析{company}的财务表现。"),
        SystemMessage(content="请分析公司的收入、利润、毛利率、净利率、现金流等关键财务指标，并评估公司的财务健康状况。"),
        SystemMessage(content="你可以使用 web_search 工具查找最新的财务报表和数据。")
    ] + state["messages"]
    
    # 生成响应
    llm = init_llm()
    llm_with_tools = ModelScopeChatWithTools(llm, [web_search])
    response = llm_with_tools.invoke(financial_analyst_prompt)
    
    return {"messages": [AIMessage(content=f"## 财务分析\n\n{response.content}")]}

def manager_agent_node(state: AgentState):
    """
    经理智能体：合成各专家的分析结果，生成最终报告
    
    参数：
        state: 智能体状态，包含所有专家的分析结果
    
    返回：
        更新后的状态，包含最终报告
    """
    log_message = "--- 经理：合成最终报告... ---"
    GLOBAL_LOGS.append(log_message)
    console = Console()
    console.print(log_message)
    
    # 提取公司名称
    company = extract_company_name(state["messages"])
    
    # 收集所有专家的分析结果
    expert_analyses = ""
    for msg in state["messages"]:
        if isinstance(msg, AIMessage) and "## " in msg.content:
            expert_analyses += msg.content + "\n\n"
    
    # 构建经理的提示
    manager_prompt = [
        SystemMessage(content="你是一位资深的投资经理，负责整合各领域专家的分析结果。"),
        SystemMessage(content="请将以下各专家的分析整合成一份结构清晰、内容全面的市场分析报告。"),
        SystemMessage(content="报告应该包括执行摘要、各部分详细分析以及最终结论。"),
        HumanMessage(content=f"请根据以下专家分析，为{company}生成一份完整的市场分析报告：\n\n{expert_analyses}")
    ]
    
    # 生成最终报告
    llm = init_llm()
    manager_response = llm.invoke("\n".join([msg.content for msg in manager_prompt]))
    
    return {"messages": [AIMessage(content=manager_response)]}

# 辅助函数
def extract_company_name(messages: List[BaseMessage]) -> str:
    """
    从消息历史中提取公司名称
    
    参数：
        messages: 消息历史列表
    
    返回：
        提取的公司名称
    """
    for msg in messages:
        if isinstance(msg, HumanMessage):
            # 查找公司名称的模式，例如 "NVIDIA (NVDA)"
            match = re.search(r'([\u4e00-\u9fa5A-Za-z]+)\s*\([A-Z]+\)', msg.content)
            if match:
                return match.group(1)
    # 默认返回 NVIDIA
    return "NVIDIA"

# =========================
# 4) 工作流构建
# =========================
def build_multi_agent_system():
    """
    构建多智能体系统工作流
    
    返回：
        编译后的 LangGraph 应用
    """
    # ========================================================
    # 1. 定义多智能体系统的共享状态
    # ========================================================
    class MultiAgentState(TypedDict):
        # 与AgentState类似，使用add_messages确保消息安全合并
        # 这意味着每个智能体的输出都会被追加到消息列表中，而不是替换
        messages: Annotated[list[AnyMessage], add_messages]
    
    # ========================================================
    # 2. 创建状态图构建器
    # ========================================================
    # StateGraph是LangGraph的核心组件，用于定义智能体工作流
    # 它维护一个共享状态，并根据定义的规则在节点之间传递
    multi_graph_builder = StateGraph(MultiAgentState)
    
    # ========================================================
    # 3. 添加智能体节点到工作流
    # ========================================================
    # 每个节点代表一个专业化的智能体，负责特定领域的分析
    #
    # 节点1："news_analyst" - 新闻分析师，分析公司的最新新闻和市场情绪
    multi_graph_builder.add_node("news_analyst", news_analyst_node)
    
    # 节点2："technical_analyst" - 技术分析师，分析公司股票的技术走势
    multi_graph_builder.add_node("technical_analyst", technical_analyst_node)
    
    # 节点3："financial_analyst" - 财务分析师，分析公司的财务表现
    multi_graph_builder.add_node("financial_analyst", financial_analyst_node)
    
    # 节点4："manager" - 经理，合成各专家的分析结果，生成最终报告
    multi_graph_builder.add_node("manager", manager_agent_node)
    
    # ========================================================
    # 4. 定义工作流的执行顺序
    # ========================================================
    # 设置工作流的入口点：从新闻分析师开始执行
    multi_graph_builder.set_entry_point("news_analyst")
    
    # 定义节点之间的执行顺序（简单的线性工作流）
    # 1. 新闻分析师完成后，转到技术分析师
    multi_graph_builder.add_edge("news_analyst", "technical_analyst")
    
    # 2. 技术分析师完成后，转到财务分析师
    multi_graph_builder.add_edge("technical_analyst", "financial_analyst")
    
    # 3. 财务分析师完成后，转到经理合成报告
    multi_graph_builder.add_edge("financial_analyst", "manager")
    
    # 4. 经理完成后，结束工作流
    multi_graph_builder.add_edge("manager", END)
    
    # ========================================================
    # 5. 编译工作流并返回
    # ========================================================
    # 编译后的工作流是一个可调用的对象，可以接收初始状态并执行
    # 执行时，它会按照定义的顺序依次调用各个智能体节点
    return multi_graph_builder.compile()

# =========================
# 5) 主函数与运行逻辑
# =========================
def main(company: str = "NVIDIA (NVDA)"):
    """
    主函数：运行单智能体和多智能体系统进行对比
    
    参数：
        company: 要分析的公司，格式为 "公司名称 (股票代码)"
    """
    # 初始化控制台
    console = Console()
    
    # 加载环境变量
    load_dotenv()
    
    # 设置环境变量
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_PROJECT"] = "Agentic Architecture - Multi-Agent (ModelScope)"
    
    # 检查环境变量
    for key in ["MODELSCOPE_API_KEY"]:
        if not os.environ.get(key):
            console.print(f"未找到{key}。请创建.env文件并设置该变量。", style="bold red")
            return
    
    console.print("环境变量已加载，追踪已设置完成。", style="bold green")
    
    # 初始化 LLM 和工具
    llm = init_llm()
    llm_with_tools = ModelScopeChatWithTools(llm, [web_search])
    
    # 构建单智能体系统
    monolithic_agent = build_monolithic_agent(llm_with_tools)
    console.print("单智能体系统编译成功。", style="bold green")
    
    # 构建多智能体系统
    multi_agent_system = build_multi_agent_system()
    console.print("多智能体系统编译成功。", style="bold green")
    
    # 定义查询
    analysis_query = f"为{company}创建一份简要但全面的市场分析报告。报告应包括三个部分：1. 最近新闻和市场情绪摘要。2. 股票价格趋势的基本技术分析。3. 公司最近财务表现的分析。"
    
    # 测试单智能体系统
    console.print(f"\n[bold yellow]测试单智能体系统：[/bold yellow]\n'{analysis_query}'\n")
    
    final_mono_output = monolithic_agent.invoke({
        "messages": [
            SystemMessage(content="你是一位专业的金融分析师。你必须创建一份全面的报告，涵盖用户请求的所有方面。"),
            HumanMessage(content=analysis_query)
        ]
    })
    
    console.print("\n--- [bold red]单智能体系统最终报告[/bold red] ---")
    console.print(Markdown(final_mono_output['messages'][-1].content))
    
    # 测试多智能体系统
    console.print(f"\n[bold yellow]\n测试多智能体系统：[/bold yellow]\n'{analysis_query}'\n")
    
    final_multi_output = multi_agent_system.invoke({
        "messages": [
            HumanMessage(content=analysis_query)
        ]
    })
    
    console.print("\n--- [bold green]多智能体系统最终报告[/bold green] ---")
    console.print(Markdown(final_multi_output['messages'][-1].content))
    
    # 比较结果
    console.print("\n=== [bold blue]系统比较[/bold blue] ===")
    console.print("单智能体系统：")
    console.print("- 优点：结构简单，单一入口")
    console.print("- 缺点：分析可能不够深入，各领域专业度有限")
    
    console.print("\n多智能体系统：")
    console.print("- 优点：各领域分析更深入，专业度更高，结果更全面")
    console.print("- 缺点：结构复杂，需要更多的协调和资源")

if __name__ == "__main__":
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="多智能体系统市场分析工具")
    parser.add_argument("--company", type=str, default="NVIDIA (NVDA)", help="要分析的公司，格式为 '公司名称 (股票代码)'")
    
    # 解析参数
    args = parser.parse_args()
    
    # 运行主函数
    main(company=args.company)
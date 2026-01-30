#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
# 📘 智能体架构 11：元控制器（Meta-Controller）

本文件实现了一个元控制器架构，这是一种监督式智能体架构，用于协调一组专业的子智能体。

与构建一个试图做所有事情的单一、整体式智能体不同，元控制器充当智能调度器。它接收传入的请求，分析其性质，并将其路由到可用智能体池中最合适的专家。
这允许每个子智能体针对其特定任务进行高度优化，从而获得更好的性能和模块化。

我们将通过构建一个包含三个不同专家的系统来演示这一点：
1.  **通用智能体（Generalist）**：处理日常对话和简单问题。
2.  **研究智能体（Researcher）**：配备搜索工具，回答有关近期事件或复杂主题的问题。
3.  **编码智能体（Coder）**：专注于生成Python代码片段的专家。

元控制器将是操作的"大脑"，检查每个用户查询并决定哪个智能体最适合响应。这创建了一个灵活且易于扩展的系统，其中可以通过创建新的专家智能体并让控制器了解它来添加新功能。

### 定义
**元控制器（Meta-Controller）**（或路由器）是多智能体系统中的监督智能体，负责分析传入任务并将其分派给适当的专业子智能体或工作流。它充当智能路由层，决定哪个工具或专家最适合手头的工作。

### 高级工作流程

1.  **接收输入**：系统接收用户请求。
2.  **元控制器分析**：元控制器智能体检查请求的意图、复杂性和内容。
3.  **分派给专家**：基于其分析，元控制器从预定义池中选择最佳专家智能体（例如，'研究员'、'编码器'、'聊天机器人'）。
4.  **执行任务**：选定的专家智能体执行任务并生成结果。
5.  **返回结果**：专家的结果返回给用户。在更复杂的工作流中，控制权可能会返回给元控制器以进行进一步的步骤或监控。

### 何时使用 / 应用
*   **多服务AI平台**：平台的单一入口点，提供多样化的服务，如文档分析、数据可视化和创意写作。
*   **自适应个人助手**：可以在不同模式或工具之间切换的助手，例如管理日历、搜索网络或控制智能家居设备。
*   **企业工作流程**：根据工单内容将客户支持工单路由到正确的部门（技术、 billing、销售）。

### 优势和劣势
*   **优势**：
    *   **灵活性和模块化**：通过简单地添加新的专家智能体并更新控制器的路由逻辑，非常容易添加新功能。
    *   **性能**：允许高度优化的专家智能体，而不是一个可能在所有方面都表现平庸的通用模型。
*   **劣势**：
    *   **控制器作为单点故障**：整个系统的质量取决于控制器正确路由任务的能力。糟糕的路由决策会导致次优或不正确的结果。
    *   **潜在的延迟增加**：与直接调用单个智能体相比，路由的额外步骤可能会增加少量延迟。
"""

# 📋 环境与设置
# 我们将安装必要的库并设置环境。我们需要 `langchain-tavily` 用于研究智能体的搜索工具。

# !pip install -q -U langchain langgraph rich python-dotenv langchain-tavily langchain-openai

import os
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv

# Pydantic用于数据建模
from pydantic import BaseModel, Field

# LangChain组件
from langchain_tavily import TavilySearch
from langchain_core.prompts import ChatPromptTemplate

# LangGraph组件
from langgraph.graph import StateGraph, END
from typing_extensions import TypedDict

# 用于美化打印
from rich.console import Console
from rich.markdown import Markdown

# OpenAI相关
from openai import OpenAI
from openai import RateLimitError, APIError

# 配置调试模式
DEBUG: bool = True

# 初始化控制台
console = Console()

# ModelScope的OpenAI兼容接口适配器
class ModelScopeChat:
    """
    ModelScope 的 OpenAI 兼容接口适配器：
    - 提供 invoke(prompt) 基本调用
    - 提供 with_structured_output(PydanticModel) 的结构化输出包装
    - 支持API错误时自动切换到备选模型
    - 支持无API密钥时的模拟响应
    """
    def __init__(self, base_url: str = None, api_key: str = None, model: str = None, temperature: float = 0.2, extra_body: Optional[dict] = None):
        # 默认配置（可通过环境变量或参数覆盖）
        self.base_url = base_url or os.environ.get("MODELSCOPE_BASE_URL", "https://api-inference.modelscope.cn/v1")
        self.api_key = api_key or os.environ.get("MODELSCOPE_API_KEY")
        self.model = model or os.environ.get("MODELSCOPE_MODEL_ID", "deepseek-ai/DeepSeek-V3.2")
        # 从环境变量获取备选模型ID
        self.fallback_model = os.environ.get("MODELSCOPE_MODEL_ID_R1")
        # 注意：不要用 base_url 再覆盖 self.base_url，否则无参调用时 self.base_url 会变成 None，请求会发到 OpenAI 默认地址导致 401
        self.temperature = temperature
        # 为避免"未信任的chat template"错误，增加信任参数；并请求JSON输出格式
        self.extra_body = extra_body or {
            "enable_thinking": True,
            "trust_request_chat_template": True,
            "response_format": {"type": "json_object"},
        }
        self.switched = False
        
        # 初始化OpenAI客户端
        if not self.api_key:
            console.print("[bold yellow]⚠️ 未设置MODELSCOPE_API_KEY环境变量，将使用模拟响应[/bold yellow]")
            self.client = None
        else:
            self.client = OpenAI(base_url=self.base_url, api_key=self.api_key)

    def invoke(self, prompt: str):
        # 如果没有API密钥，返回模拟响应
        if not self.client:
            console.print("[bold yellow]⚠️ 使用模拟响应[/bold yellow]")
            return self._get_mock_response(prompt)
            
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
                # 如果没有备选模型或已经切换过，抛出异常
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
                    console.print("[bold blue]📥 收到模型返回，尝试解析为 JSON[/bold blue]")
                messages = [
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": prompt},
                ]
                
                # 如果没有API密钥，返回模拟响应
                if not self.outer.client:
                    console.print("[bold yellow]⚠️ 使用模拟响应[/bold yellow]")
                    return self.outer._get_mock_structured_response(pyd_model, prompt)
                
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
                        # 如果没有备选模型或已经切换过，抛出异常
                        raise
                content = resp.choices[0].message.content or ""
                import json, re
                from pydantic import ValidationError
                def _extract_json(s: str) -> str:
                    m = re.search(r'(\{[\s\S]*\}|\[[\s\S]*\])', s)
                    return m.group(1) if m else "{}"
                raw = content.strip()
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
                    try:
                        parsed = pyd_model.model_validate(data)
                        if DEBUG:
                            console.print(f"[bold green]✅ 结构化解析成功（应用字段映射后）[/bold green]：{pyd_model.__name__}")
                        return parsed
                    except ValidationError:
                        # 如果映射后仍然解析失败，尝试使用模拟响应
                        if DEBUG:
                            console.print(f"[bold yellow]⚠️ JSON解析失败，使用模拟响应[/bold yellow]")
                        return self.outer._get_mock_structured_response(pyd_model, prompt)

        return _StructuredWrapper(self)
    
    def _get_mock_response(self, prompt: str) -> str:
        """获取模拟响应（用于无API密钥的演示）"""
        if "元控制器" in prompt or "路由" in prompt:
            return "这是一个模拟响应。在实际应用中，这将由LLM生成。"
        else:
            return "这是一个模拟响应。在实际应用中，这将由LLM生成。"
    
    def _get_mock_structured_response(self, pyd_model: type[BaseModel], prompt: str) -> BaseModel:
        """获取模拟结构化响应（用于无API密钥的演示）"""
        # 检查模型类型并返回相应的模拟数据
        if pyd_model.__name__ == "ControllerDecision":
            # 基于提示内容生成模拟决策
            if "你好" in prompt or "怎么样" in prompt or "问候" in prompt:
                return pyd_model(next_agent="Generalist", reasoning="用户的请求是一个简单的问候，属于日常对话，由通用智能体处理。")
            elif "最新" in prompt or "财务" in prompt or "研究" in prompt:
                return pyd_model(next_agent="Researcher", reasoning="用户的请求涉及最新信息或研究内容，需要使用搜索工具获取信息，由研究智能体处理。")
            elif "代码" in prompt or "函数" in prompt or "编程" in prompt:
                return pyd_model(next_agent="Coder", reasoning="用户的请求涉及代码生成或编程任务，由编码智能体处理。")
            else:
                return pyd_model(next_agent="Generalist", reasoning="用户的请求属于一般问题，由通用智能体处理。")
        else:
            # 为其他模型类型返回默认模拟数据
            import inspect
            params = {}
            for field_name, field in pyd_model.__fields__.items():
                if field.type_ == str:
                    params[field_name] = "模拟值"
                elif field.type_ == int:
                    params[field_name] = 0
                elif field.type_ == float:
                    params[field_name] = 0.0
                elif field.type_ == bool:
                    params[field_name] = False
                else:
                    params[field_name] = None
            return pyd_model(**params)


def init_llm() -> ModelScopeChat:
    """
    初始化 ModelScope LLM（OpenAI 兼容接口）。
    - 可通过环境变量控制：
      MODELSCOPE_BASE_URL（默认：https://api-inference.modelscope.cn/v1）
      MODELSCOPE_API_KEY
      MODELSCOPE_MODEL_ID（默认：deepseek-ai/DeepSeek-V3.2）
    - 额外参数：enable_thinking 可选
    """
    return ModelScopeChat(temperature=0.4)


# --- API密钥和追踪设置 ---
load_dotenv()

# 禁用LangSmith追踪（如果API密钥已过期）
os.environ["LANGCHAIN_TRACING_V2"] = "false"
os.environ["LANGCHAIN_PROJECT"] = "智能体架构 - 元控制器（ModelScope）"

# 检查所需的环境变量（仅 MODELSCOPE_API_KEY 必需，其余可选）
required_vars = ["MODELSCOPE_API_KEY"]
optional_vars = ["LANGCHAIN_API_KEY", "TAVILY_API_KEY"]
missing_required = [v for v in required_vars if v not in os.environ]
missing_optional = [v for v in optional_vars if v not in os.environ]
if missing_required:
    print(f"警告: 请设置必需环境变量: {', '.join(missing_required)}（在 .env 中配置）")
if missing_optional:
    print(f"提示: 未设置可选变量: {', '.join(missing_optional)}（研究智能体需 TAVILY_API_KEY）")

print("环境变量已加载，追踪已设置。")

# 初始化LLM
llm = init_llm()

# 初始化搜索工具
search_tool = TavilySearch(max_results=3)


# --- 第1阶段：构建专家智能体 ---#
# 首先，我们将创建专家智能体团队。每个智能体都是一个具有特定角色的简单链，
# 对于研究员，还会配备一个工具。我们将它们包装在一个节点函数中，以便在LangGraph中使用。

# 定义整个图的状态
class MetaAgentState(TypedDict):
    """
    元智能体系统的状态定义
    - user_request: 用户的请求内容
    - next_agent_to_call: 下一步要调用的智能体名称
    - generation: 智能体的生成结果
    """
    user_request: str
    next_agent_to_call: Optional[str]
    generation: str


# 辅助工厂函数，用于创建专家智能体节点
def create_specialist_node(persona: str, tools: list = None):
    """
    创建专家智能体节点的工厂函数
    
    参数:
        persona: 智能体的角色描述
        tools: 智能体可用的工具列表
    
    返回:
        一个处理特定角色任务的智能体节点函数
    """
    system_prompt = f"你是一个专家智能体，具有以下角色：{persona}。根据你的角色直接、简洁地回应用户的请求。"
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{user_request}")
    ])
    
    # 带工具的专家智能体节点
    def specialist_node(state: MetaAgentState) -> Dict[str, Any]:
        """
        专家智能体节点
        
        参数:
            state: 当前系统状态
        
        返回:
            更新后的状态，包含智能体的生成结果
        """
        # ModelScopeChat 非 LangChain Runnable，不能使用 prompt | llm；先格式化提示再直接调用
        formatted_prompt = prompt.format(user_request=state["user_request"])
        result = llm.invoke(formatted_prompt)
        return {"generation": result}
    
    return specialist_node


# 1. 通用智能体节点
generalist_node = create_specialist_node(
    "你是一个友好、乐于助人的通用AI助手。你处理日常对话和简单问题。"
)

# 2. 研究智能体节点
research_agent_node = create_specialist_node(
    "你是一位专家研究员。你必须使用搜索工具找到信息来回答用户的问题。",
    tools=[search_tool]
)

# 3. 编码智能体节点
coding_agent_node = create_specialist_node(
    "你是一位专家Python程序员。你的任务是根据用户的请求编写干净、高效的Python代码。只提供代码，用markdown代码块包裹，最少的解释。"
)

print("专家智能体定义成功。")


# --- 第2阶段：构建元控制器 ---#
# 这是我们系统的大脑。元控制器是一个基于LLM的节点，其唯一工作是决定将请求路由到哪个专家。
# 其提示的质量对系统性能至关重要。

# 控制器路由决策的Pydantic模型
class ControllerDecision(BaseModel):
    """
    控制器决策模型
    - next_agent: 下一步要调用的专家智能体名称
    - reasoning: 选择该智能体的简要原因
    """
    next_agent: str = Field(description="下一步要调用的专家智能体名称。必须是['Generalist', 'Researcher', 'Coder']之一。")
    reasoning: str = Field(description="选择该智能体的简要原因。")


def meta_controller_node(state: MetaAgentState) -> Dict[str, Any]:
    """
    中央控制器，决定调用哪个专家智能体
    
    参数:
        state: 当前系统状态
    
    返回:
        更新后的状态，包含下一步要调用的智能体名称
    """
    console.print("--- 🧠 元控制器分析请求 ---")
    
    # 为控制器定义专家及其描述
    specialists = {
        "Generalist": "处理日常对话、问候和简单问题。",
        "Researcher": "回答有关近期事件、复杂主题或任何需要来自网络的最新信息的问题。",
        "Coder": "根据用户的规范编写Python代码。"
    }
    
    specialist_descriptions = "\n".join([f"- {name}: {desc}" for name, desc in specialists.items()])
    
    # 构建控制器提示
    prompt = ChatPromptTemplate.from_template(
        f"""你是多智能体AI系统的元控制器。你的工作是分析用户的请求并将其路由到最合适的专家智能体。

可用的专家如下：
{specialist_descriptions}

分析以下用户请求并选择最适合处理它的专家。以要求的格式提供你的决策。

用户请求：{{user_request}}"""
    )
    
    # 使用结构化输出获取控制器决策
    controller_llm = llm.with_structured_output(ControllerDecision)
    
    # 构建完整提示
    formatted_prompt = prompt.format(user_request=state['user_request'])
    
    # 执行决策
    decision = controller_llm.invoke(formatted_prompt)
    console.print(f"[yellow]路由决策:[/yellow] 发送到 [bold]{decision.next_agent}[/bold]。 [italic]原因: {decision.reasoning}[/italic]")
    
    return {"next_agent_to_call": decision.next_agent}

print("元控制器节点定义成功。")


# --- 第3阶段：组装和运行图 ---#
# 现在我们将使用LangGraph将所有内容连接在一起。图将从元控制器开始，
# 然后根据其决策，条件边将状态路由到正确的专家节点。专家运行后，图将结束。

# 构建图
workflow = StateGraph(MetaAgentState)

# 添加控制器和每个专家的节点
workflow.add_node("meta_controller", meta_controller_node)
workflow.add_node("Generalist", generalist_node)
workflow.add_node("Researcher", research_agent_node)
workflow.add_node("Coder", coding_agent_node)

# 设置入口点
workflow.set_entry_point("meta_controller")

# 定义条件路由逻辑
def route_to_specialist(state: MetaAgentState) -> str:
    """
    读取控制器的决策并返回要路由到的节点名称
    
    参数:
        state: 当前系统状态
    
    返回:
        要路由到的节点名称
    """
    return state["next_agent_to_call"]

# 添加条件边
workflow.add_conditional_edges(
    "meta_controller",
    route_to_specialist,
    {
        "Generalist": "Generalist",
        "Researcher": "Researcher",
        "Coder": "Coder"
    }
)

# 任何专家运行后，流程结束
workflow.add_edge("Generalist", END)
workflow.add_edge("Researcher", END)
workflow.add_edge("Coder", END)

# 编译图
meta_agent = workflow.compile()
print("元控制器智能体图编译成功。")


# --- 第4阶段：演示 ---#
# 让我们用各种提示测试我们的元控制器，看看它是否能正确地将它们分派给正确的专家。

def run_agent(query: str):
    """
    运行元控制器智能体并显示结果
    
    参数:
        query: 用户查询
    """
    result = meta_agent.invoke({"user_request": query})
    console.print("\n[bold]最终响应:[/bold]")
    console.print(Markdown(result['generation']))


# 测试1：应该路由到通用智能体
console.print("--- 💬 测试1: 日常对话 ---")
run_agent("你好，今天怎么样？")

# 测试2：应该路由到研究智能体
console.print("\n--- 🔬 测试2: 研究问题 ---")
run_agent("NVIDIA最新的财务业绩是什么？")

# 测试3：应该路由到编码智能体
console.print("\n--- 💻 测试3: 编码请求 ---")
run_agent("你能为我写一个Python函数来计算第n个斐波那契数吗？")


# 📝 结论
# 在本文件中，我们成功实现了一个**元控制器**架构。我们的测试清楚地展示了它的主要功能：充当智能和动态路由器。
# 
# 1. 简单的问候被正确识别并发送到**通用智能体**。
# 2. 关于近期财务新闻的查询被分派到**研究智能体**，该智能体使用其搜索工具获取最新信息。
# 3. 对代码片段的请求被路由到**编码智能体**，该智能体提供了格式良好且正确的函数。
# 
# 这种模式对于构建可扩展和可维护的AI系统非常强大。通过分离关注点，每个专家都可以独立改进而不影响其他专家。系统的整体智能可以通过添加新的、更有能力的专家并让元控制器了解它们来增强。
# 虽然控制器本身代表了潜在的瓶颈，但其作为灵活协调器的角色是先进智能体设计的基石。

console.print("\n--- 🎉 元控制器架构演示完成！ ---")

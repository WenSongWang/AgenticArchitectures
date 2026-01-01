# 📘 Agentic Architectures 4: Planning

# 本脚本探索规划架构，并将其与ReAct模型进行比较。
# 规划智能体在采取任何行动之前，首先将复杂任务分解为子目标，
# 为多步骤任务提供结构和效率。

# 阶段 0：基础与设置
# 安装必要的库：
# !pip install -q -U langchain-nebius langchain langgraph rich python-dotenv langchain-tavily

import os
import re
import json
from typing import List, Annotated, TypedDict, Optional
from dotenv import load_dotenv

# LangChain components
from langchain_core.messages import BaseMessage, ToolMessage, SystemMessage, HumanMessage, AIMessage
from pydantic import BaseModel, Field
from langchain_core.tools import tool
from langchain_tavily import TavilySearch
from openai import OpenAI, RateLimitError, APIError
import logging
from rich.logging import RichHandler

# LangGraph components
from langgraph.graph import StateGraph, END
from langgraph.graph.message import AnyMessage, add_messages
from langgraph.prebuilt import ToolNode, tools_condition

# For pretty printing
from rich.console import Console
from rich.markdown import Markdown

# --- API Key and Tracing Setup ---
load_dotenv()

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "Agentic Architecture - Planning (ModelScope)"

# Check that the keys are set
for key in ["MODELSCOPE_API_KEY", "LANGCHAIN_API_KEY", "TAVILY_API_KEY"]:
    if not os.environ.get(key):
        print(f"未找到{key}。请创建.env文件并设置该变量。")

print("环境变量已加载，追踪已设置完成。")

# Define ModelScopeChat class for compatibility
class ModelScopeChat:
    """
    ModelScope 的 OpenAI 兼容接口适配器
    - 提供 invoke(prompt) 基本调用
    - 提供 with_structured_output(PydanticModel) 的结构化输出包装
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
                return resp.choices[0].message.content
            else:
                raise

    def with_structured_output(self, pyd_model: type[BaseModel]):
        class _StructuredWrapper:
            def __init__(self, outer: "ModelScopeChat"):
                self.outer = outer

            def invoke(self, prompt: str) -> BaseModel:
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
                # 兜底字段映射
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
    base_url = os.environ.get("MODELSCOPE_BASE_URL", "https://api-inference.modelscope.cn/v1")
    api_key = os.environ.get("MODELSCOPE_API_KEY", "")
    model_id = os.environ.get("MODELSCOPE_MODEL_ID", "deepseek-ai/DeepSeek-V3.2")
    fallback_model_id = os.environ.get("MODELSCOPE_MODEL_ID_R1")
    # 为避免“未信任的 chat template”错误，增加信任参数；并请求 JSON 输出格式
    extra = {
        "enable_thinking": True,
        "trust_request_chat_template": True,
        "response_format": {"type": "json_object"},
    }
    return ModelScopeChat(base_url=base_url, api_key=api_key, model=model_id, fallback_model=fallback_model_id, temperature=0.2, extra_body=extra)

# Phase 1: The Baseline - A Reactive Agent (ReAct)
# We'll rebuild the ReAct agent to compare against the planning agent

console = Console()

# 3. Define the state for our graphs
class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]

# 简单的模拟网络搜索工具用于测试
@tool
def web_search(query: str) -> str:
    """模拟网络搜索并返回模拟结果。"""
    console.print(f"--- 工具：搜索 '{query}'...")
    
    # 常见查询的模拟结果
    mock_data = {
        "北京人口": "北京的人口约为2154万（2023年）。",
        "上海人口": "上海的人口约为2487万（2023年）。",
        "广州人口": "广州的人口约为1873万（2023年）。",
        "深圳人口": "深圳的人口约为1756万（2023年）。",
        "中国人口": "中国的人口约为14.12亿（2023年）。"
    }
    
    # 返回模拟数据或通用响应
    for key, value in mock_data.items():
        if key in query:
            return value
    
    return f"模拟搜索结果：{query}"

# 3. Define the LLM and bind it to our custom tool
llm = init_llm()

# Create a wrapper for ModelScopeChat to work with bind_tools
class ModelScopeChatWithTools:
    def __init__(self, llm_instance: ModelScopeChat, tools: list):
        self.llm = llm_instance
        self.tools = tools
    
    def invoke(self, messages: list):
        # Convert messages to a single prompt string
        prompt = ""
        for msg in messages:
            if isinstance(msg, SystemMessage):
                prompt += f"[SYSTEM] {msg.content}\n"
            elif isinstance(msg, HumanMessage):
                prompt += f"[USER] {msg.content}\n"
            elif isinstance(msg, AIMessage):
                prompt += f"[ASSISTANT] {msg.content}\n"
            else:
                # For other message types, try to extract content
                try:
                    prompt += f"[MESSAGE] {msg.content}\n"
                except:
                    continue
        
        # Generate response
        response = self.llm.invoke(prompt)
        
        # Create a response object that extends LangChain's AIMessage
        from langchain_core.messages import AIMessage
        
        # Parse tool calls if any
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
                # If parsing fails, check for simple pattern
                import re
                match = re.search(r'web_search\(query=[\'"]([^\'"]+)[\'"]\)', response)
                if match:
                    tool_calls = [{
                        "name": "web_search",
                        "args": {"query": match.group(1)}
                    }]
        
        # Create and return an actual AIMessage object
        return AIMessage(content=response, tool_calls=tool_calls)

# Use the wrapper with tools
llm_with_tools = ModelScopeChatWithTools(llm, [web_search])

# 4. Agent node with a system prompt to force one tool call at a time
def react_agent_node(state: AgentState):
    console.print("--- 反应式智能体：思考中... ---")
    
    messages_with_system_prompt = [
        SystemMessage(content="你是一个乐于助人的研究助手。每次必须且只能调用一个工具。不要在一次对话中调用多个工具。收到工具结果后，你将决定下一步操作。")
    ] + state["messages"]

    response = llm_with_tools.invoke(messages_with_system_prompt)
    
    return {"messages": [response]}

# 5. Use our custom tool in the ToolNode
tool_node = ToolNode([web_search])

# The ReAct graph with its characteristic loop
react_graph_builder = StateGraph(AgentState)
react_graph_builder.add_node("agent", react_agent_node)
react_graph_builder.add_node("tools", tool_node)
react_graph_builder.set_entry_point("agent")
react_graph_builder.add_conditional_edges("agent", tools_condition)
react_graph_builder.add_edge("tools", "agent")

react_agent_app = react_graph_builder.compile()
print("反应式(ReAct)智能体编译成功。")

# 在以规划为中心的查询上测试反应式智能体
print("\n在以规划为中心的查询上测试反应式智能体：")
print("'查找北京、上海和广州的人口。 ")
print("然后计算它们的总人口。 ")
print("最后，将总人口与中国人口进行比较，并说明哪个更大。'")

react_result = react_agent_app.invoke({
    "messages": [
        ("human", "查找北京、上海和广州的人口。 "
                 "然后计算它们的总人口。 "
                 "最后，将总人口与中国人口进行比较，并说明哪个更大。")
    ]
})

# 阶段2：规划智能体
# 现在我们将构建一个规划智能体，它会在执行前创建一个逐步计划

# 定义计划的模式
plan_schema_prompt = """
我需要你创建一个解决此问题的详细计划。你的计划应该是一个编号的具体步骤列表。
每个步骤应该是一个简单的任务，可以通过单次工具调用或简单计算完成。

示例：
1. 搜索北京的人口。
2. 搜索上海的人口。
3. 搜索广州的人口。
4. 将这三个人口相加，得到总人口。
5. 搜索中国的人口。
6. 将总人口与中国人口进行比较，并说明哪个更大。

请以编号列表的形式提供你的计划，不要添加任何额外的解释。
"""

# Planner node - creates a detailed plan before execution
def planner_node(state: AgentState):
    console.print("--- 规划器：创建计划中... ---")
    
    # Add the plan schema prompt to the messages
    plan_prompt = [SystemMessage(content=plan_schema_prompt)] + state["messages"]
    
    # Convert messages to prompt string
    prompt_str = ""
    for msg in plan_prompt:
        if isinstance(msg, SystemMessage):
            prompt_str += f"[SYSTEM] {msg.content}\n"
        elif isinstance(msg, HumanMessage):
            prompt_str += f"[USER] {msg.content}\n"
        elif isinstance(msg, AIMessage):
            prompt_str += f"[ASSISTANT] {msg.content}\n"
        else:
            try:
                prompt_str += f"[MESSAGE] {msg.content}\n"
            except:
                continue
    
    # Generate the plan using the LLM
    plan_response = llm.invoke(prompt_str)
    
    return {"messages": [AIMessage(content=plan_response)]}

# Plan execution node - executes the plan step by step
def executor_node(state: AgentState):
    console.print("--- 执行器：执行计划中... ---")
    
    # 获取最后一条消息（应该是计划）
    plan = state["messages"][-1].content
    
    # 从计划中提取步骤
    steps = re.findall(r"\d+\.\s*(.+)", plan)
    
    # 执行每个步骤
    execution_results = []
    for step in steps:
        console.print(f"\n--- 执行步骤：{step} ---")
        
        # Create a message for this step
        step_message = [("human", step)]
        
        # Execute the step using the reactive agent
        step_result = react_agent_app.invoke({"messages": step_message})
        
        # Add the result to our execution results
        execution_results.append(step_result["messages"][-1].content)
    
    # Return the execution results
    return {"messages": [("assistant", f"执行完成。结果：\n{chr(10).join(execution_results)}")]}

# Synthesis node - combines the results into a final answer
def synthesizer_node(state: AgentState):
    console.print("--- 合成器：创建最终答案中... ---")
    
    # 获取所有消息
    messages = state["messages"]
    
    # 创建合成提示
    synthesis_prompt_list = [SystemMessage(content="你是一个乐于助人的助手。请根据上述执行结果提供清晰简洁的最终答案。")] + messages
    
    # Convert messages to prompt string
    synthesis_prompt_str = ""
    for msg in synthesis_prompt_list:
        if isinstance(msg, SystemMessage):
            synthesis_prompt_str += f"[SYSTEM] {msg.content}\n"
        elif isinstance(msg, HumanMessage):
            synthesis_prompt_str += f"[USER] {msg.content}\n"
        elif isinstance(msg, AIMessage):
            synthesis_prompt_str += f"[ASSISTANT] {msg.content}\n"
        else:
            try:
                synthesis_prompt_str += f"[MESSAGE] {msg.content}\n"
            except:
                continue
    
    # Generate the final answer
    final_answer = llm.invoke(synthesis_prompt_str)
    
    return {"messages": [AIMessage(content=final_answer)]}

# Create the planning agent graph
planning_graph_builder = StateGraph(AgentState)
planning_graph_builder.add_node("planner", planner_node)
planning_graph_builder.add_node("executor", executor_node)
planning_graph_builder.add_node("synthesizer", synthesizer_node)

# Define the flow
planning_graph_builder.set_entry_point("planner")
planning_graph_builder.add_edge("planner", "executor")
planning_graph_builder.add_edge("executor", "synthesizer")
planning_graph_builder.add_edge("synthesizer", END)

# 编译图
planning_agent_app = planning_graph_builder.compile()
print("规划智能体编译成功。")

if __name__ == "__main__":
    # 测试规划智能体
    print("\n在相同查询上测试规划智能体：")
    print("'查找北京、上海和广州的人口。 ")
    print("然后计算它们的总人口。 ")
    print("最后，将总人口与中国人口进行比较，并说明哪个更大。'")
    
    planning_result = planning_agent_app.invoke({
        "messages": [
            ("human", "查找北京、上海和广州的人口。 "
                     "然后计算它们的总人口。 "
                     "最后，将总人口与中国人口进行比较，并说明哪个更大。")
        ]
    })
    
    # 比较结果
    print("\n=== 比较结果 ===")
    print("反应式(ReAct)智能体结果：")
    print(react_result["messages"][-1].content)
    print("\n规划智能体结果：")
    print(planning_result["messages"][-1].content)
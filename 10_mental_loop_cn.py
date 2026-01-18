#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
# 📘 智能体架构 10：模拟器 / 思维模型循环 (Simulator / Mental-Model-in-the-Loop)

一种为高风险环境设计的复杂架构：**模拟器**，也称为**思维模型循环**。

核心思想是赋予智能体"先思考后行动"的能力。智能体在现实世界中采取行动之前，首先会在内部模拟环境中测试其提议的行动。通过在这个安全的沙箱中观察可能的结果，它可以评估风险、完善策略，然后再在现实中执行更周全的行动。

我将构建一个简单的**股票交易智能体**来演示这一点。"现实世界"将是一个市场模拟器，每次推进一个步骤。在进行交易之前，我们的智能体将：
1. 提出一个通用策略（例如，"激进买入"）。
2. 在市场模拟器的*分叉*版本中运行该策略多个未来步骤，查看潜在结果。
3. 分析模拟结果，评估风险和回报。
4. 做出最终的、经过改进的决策（例如，"模拟显示波动性很大；让我们少买一点"）。
5. 在真实市场中执行这种经过改进的交易。

这种模式对于将智能体从信息任务转移到现实世界中的行动至关重要，因为在现实世界中，错误可能会产生实际后果。

### 定义
**模拟器**或**思维模型循环**架构涉及一个智能体，该智能体使用其环境的内部模型，在执行任何行动之前模拟潜在行动的结果。这允许智能体执行假设分析，预测后果，并完善其计划以确保安全性和有效性。

### 高级工作流程

1.  **观察**：智能体观察真实环境的当前状态。
2.  **提出行动**：基于其目标和当前状态，智能体的规划模块生成高级提议的行动或策略。
3.  **模拟**：智能体将环境的当前状态分叉到沙盒模拟中。它应用提议的行动并运行模拟以观察一系列可能的结果。
4.  **评估与完善**：智能体分析模拟结果。该行动是否导致了期望的结果？是否有未预见的负面后果？基于此评估，它将初始提议完善为最终的、具体的行动。
5.  **执行**：智能体在*真实*环境中执行最终的、完善的行动。
6.  **重复**：循环从真实环境的新状态重新开始。

### 何时使用 / 应用
*   **机器人技术**：在移动物理手臂之前模拟抓取或路径，以避免碰撞或损坏。
*   **高风险决策**：在金融领域，模拟交易在不同市场条件下对投资组合的影响。在医疗保健领域，模拟治疗计划的潜在效果。
*   **复杂游戏AI**：策略游戏中的AI模拟未来几步以选择最优行动。

### 优势和劣势
*   **优势**：
    *   **安全性与风险降低**：通过在安全环境中预先验证行动，大大降低了有害或昂贵错误的可能性。
    *   **性能提升**：通过允许前瞻和规划，导致更稳健和考虑周全的决策。
*   **劣势**：
    *   **模拟-现实差距**：有效性完全取决于模拟器的保真度。如果世界模型不准确，智能体的计划可能基于错误的假设。
    *   **计算成本**：运行模拟，尤其是多个场景，计算成本很高，且比直接行动慢。
"""

# 📋 环境与设置
# 我们将安装所有必要的库，并配置我们的API密钥。

# !pip install -q -U langchain langgraph rich python-dotenv numpy openai

import os
import random
import numpy as np
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv

# Pydantic用于数据建模
from pydantic import BaseModel, Field

# OpenAI相关
from openai import OpenAI
from openai import RateLimitError, APIError

# LangChain组件
from langchain_core.prompts import ChatPromptTemplate

# LangGraph组件
from langgraph.graph import StateGraph, END
from typing_extensions import TypedDict

# 用于美化打印
from rich.console import Console
from rich.markdown import Markdown
from rich.table import Table

# --- API Key and Tracing Setup ---
load_dotenv()

# 禁用LangSmith追踪（解决API密钥过期问题）
os.environ["LANGCHAIN_TRACING_V2"] = "false"

# 配置调试模式
DEBUG: bool = True

# 初始化日志
import logging
from rich.logging import RichHandler

logger = logging.getLogger("mental_loop")
handler = RichHandler(console=Console(), rich_tracebacks=True, markup=True)
handler.setFormatter(logging.Formatter("%(message)s"))
logger.handlers = [handler]
logger.propagate = False
logger.setLevel(logging.DEBUG if DEBUG else logging.INFO)

console = Console()

# --- ModelScope Chat API Adapter ---
# 适配ModelScope的OpenAI兼容接口，支持中文环境和备选模型切换

class ModelScopeChat:
    """
    ModelScope 的 OpenAI 兼容接口适配器：
    - 提供 invoke(prompt) 基本调用
    - 提供 with_structured_output(PydanticModel) 的结构化输出包装
    - 支持API错误时自动切换到备选模型
    """
    def __init__(self, base_url: str = None, api_key: str = None, model: str = None, temperature: float = 0.2, extra_body: Optional[dict] = None):
        # 默认配置（可通过环境变量或参数覆盖）
        self.base_url = base_url or os.environ.get("MODELSCOPE_BASE_URL", "https://api-inference.modelscope.cn/v1")
        self.api_key = api_key or os.environ.get("MODELSCOPE_API_KEY")
        self.model = model or os.environ.get("MODELSCOPE_MODEL_ID", "deepseek-ai/DeepSeek-V3.2")
        # 从环境变量获取备选模型ID
        self.fallback_model = os.environ.get("MODELSCOPE_MODEL_ID_R1", "qwen-plus")
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
                    logger.debug("🔧 生成动态系统提示（包含字段与类型要求）")
                    logger.debug("结构化输出提示（系统消息）：\n" + system_msg)
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
                
                # 解析响应内容为JSON并转换为Pydantic模型
                import json, re
                from pydantic import ValidationError
                
                def _extract_json(s: str) -> str:
                    m = re.search(r'(\{[\s\S]*\}|\[[\s\S]*\])', s)
                    return m.group(1) if m else "{}"
                
                content = resp.choices[0].message.content or ""
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
                    
                    # ProposedAction: 常见返回别名映射
                    if "strategy" not in data and "action" in data:
                        data["strategy"] = data.pop("action")
                        mappings_applied.append("action → strategy")
                    if "reasoning" not in data and "reason" in data:
                        data["reasoning"] = data.pop("reason")
                        mappings_applied.append("reason → reasoning")
                    
                    # FinalDecision: 常见返回别名映射
                    if "action" not in data and "decision" in data:
                        data["action"] = data.pop("decision")
                        mappings_applied.append("decision → action")
                    if "amount" not in data and "quantity" in data:
                        data["amount"] = data.pop("quantity")
                        mappings_applied.append("quantity → amount")
                    
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
                            logger.error("❌ JSON解析失败，即使尝试字段映射后也无法解析，响应内容：" + content)
                        console.print(f"[bold yellow]⚠️ JSON解析失败，使用模拟响应[/bold yellow]")
                        return self.outer._get_mock_structured_response(pyd_model, prompt)
        
        return _StructuredWrapper(self)
    
    def _get_mock_response(self, prompt: str) -> str:
        """获取模拟响应（用于无API密钥的演示）"""
        if "交易策略" in prompt or "金融分析师" in prompt:
            if "正面收益" in prompt or "好消息" in prompt:
                return "激进买入。理由：正面收益报告是一个强烈的看涨信号，市场已经稳定。这是一个在价格可能进一步上涨之前进入头寸的好机会。"
            elif "新竞争对手" in prompt or "坏消息" in prompt:
                return "谨慎卖出。理由：新竞争对手的进入引入了重大不确定性和潜在的下行风险。虽然价格尚未大幅下跌，但谨慎减少敞口是明智的。"
            else:
                return "观望。理由：市场目前稳定，没有明确的看涨或看跌信号。最好等待更多信息再做出决策。"
        else:
            return "这是一个模拟响应。在实际应用中，这将由LLM生成。"
    
    def _get_mock_structured_response(self, pyd_model: type[BaseModel], prompt: str) -> BaseModel:
        """获取模拟结构化响应（用于无API密钥的演示）"""
        # 检查模型类型并返回相应的模拟数据
        if pyd_model.__name__ == "ProposedAction":
            if "正面收益" in prompt or "好消息" in prompt:
                return pyd_model(strategy="激进买入", reasoning="正面收益报告是一个强烈的看涨信号，市场已经稳定。这是一个在价格可能进一步上涨之前进入头寸的好机会。")
            elif "新竞争对手" in prompt or "坏消息" in prompt:
                return pyd_model(strategy="谨慎卖出", reasoning="新竞争对手的进入引入了重大不确定性和潜在的下行风险。虽然价格尚未大幅下跌，但谨慎减少敞口是明智的。")
            else:
                return pyd_model(strategy="观望", reasoning="市场目前稳定，没有明确的看涨或看跌信号。最好等待更多信息再做出决策。")
        elif pyd_model.__name__ == "FinalDecision":
            if "激进买入" in prompt.lower():
                return pyd_model(action="buy", amount=20.0, reasoning="模拟显示强劲的上升趋势，所有场景都产生正回报，验证了分析师的提议。我将执行20股的大量但不过度的购买，以利用预期的价格上涨，同时保持现金储备。")
            elif "谨慎卖出" in prompt.lower():
                return pyd_model(action="sell", amount=5.0, reasoning="模拟显示高度方差和负平均回报，证实了分析师的担忧。分析师谨慎卖出的初始提议是合理的。我将通过卖出5股（仓位的25%）来降低投资组合风险，锁定一些现金，并减少新竞争对手带来的潜在下行风险。")
            else:
                return pyd_model(action="hold", amount=0.0, reasoning="模拟结果显示市场将保持稳定，没有明显的上涨或下跌趋势。因此，我决定维持当前仓位。")
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

# --- 构建模拟器环境 ---

#投资组合管理模型 ，用于跟踪和计算股票交易智能体的资产状况。
class Portfolio(BaseModel):
    cash: float = 10000.0 # 当前持有的现金余额，初始值10000元
    shares: int = 0 # 当前持有的股票数量，初始值0股
    
    def value(self, current_price: float) -> float:
        """计算当前资产价值（现金+股票价值）"""
        return self.cash + self.shares * current_price

# 市场模拟器模型 ，用于模拟股票价格的变化和交易执行。
class MarketSimulator(BaseModel):
    """一个简单的单一资产股票市场模拟器。"""
    day: int = 0
    price: float = 100.0
    volatility: float = 0.1 # 价格变化的标准偏差
    drift: float = 0.01 # 一般趋势
    market_news: str = "市场稳定。"
    portfolio: Portfolio = Field(default_factory=Portfolio)

    def _analyze_news_impact(self):
        """使用LLM分析新闻对市场的影响"""
        # 构建分析提示
        prompt = f"""
请分析以下新闻对股票市场的影响程度和方向：
"{self.market_news}"

请以JSON格式返回分析结果，包含以下字段：
- impact_score: float (影响程度，范围-1到1，负数表示负面，正数表示正面)
- confidence: float (置信度，范围0到1)
- reasoning: string (简要分析理由)

示例输出：
{{
  "impact_score": 0.8,
  "confidence": 0.9,
  "reasoning": "新闻提到公司将发布积极的收益报告，预计将推动股价上涨"
}}
"""
    
        try:
            from pydantic import BaseModel
            
            class NewsImpact(BaseModel):
                impact_score: float
                confidence: float
                reasoning: str
            
            # 使用LLM进行分析
            news_llm = ModelScopeChat(temperature=0.2).with_structured_output(NewsImpact)
            result = news_llm.invoke(prompt)
            
            # 根据影响评分调整市场趋势
            if result.impact_score > 0.5:
                self.drift = 0.05  # 强正面影响
            elif result.impact_score > 0:
                self.drift = 0.02  # 弱正面影响
            elif result.impact_score < -0.5:
                self.drift = -0.05  # 强负面影响
            elif result.impact_score < 0:
                self.drift = -0.02  # 弱负面影响
            else:
                self.drift = 0.01  # 无明显影响
                
            if DEBUG:
                console.print(f"[green]📰 新闻分析结果:[/green] 影响评分={result.impact_score}, 置信度={result.confidence}")
                console.print(f"[green]💡 分析理由:[/green] {result.reasoning}")
                
        except Exception as e:
            if DEBUG:
                console.print(f"[yellow]⚠️ 新闻分析失败，使用默认逻辑: {str(e)}[/yellow]")
            # 如果LLM分析失败，回退到原始的关键词判断逻辑
            if "积极" in self.market_news or "强劲" in self.market_news:
                self.drift = 0.05
            else:
                self.drift = -0.05

    def step(self, action: str, amount: float = 0.0):
        """执行交易，然后推进模拟一天。"""
        # 1. 执行交易
        if action == "buy": # amount是股票数量
            shares_to_buy = int(amount)# 计算要购买的股票数量（取整）
            cost = shares_to_buy * self.price # 计算购买成本（股票数量*当前价格）
            if self.portfolio.cash >= cost:# 检查是否有足够的现金购买股票
                self.portfolio.shares += shares_to_buy # 更新股票数量（增加购买的股票数量）
                self.portfolio.cash -= cost # 更新现金余额（减少购买成本）
        elif action == "sell": # amount是股票数量
            shares_to_sell = int(amount)# 计算要卖出的股票数量（取整）
            if self.portfolio.shares >= shares_to_sell:# 检查是否有足够的股票卖出
                self.portfolio.shares -= shares_to_sell # 更新股票数量（减少卖出的股票数量）
                self.portfolio.cash += shares_to_sell * self.price # 更新现金余额（增加卖出股票的价值） 
        
        # 2. 更新市场价格（几何布朗运动）,根据当前趋势和波动率模拟价格变化,确保价格保持在合理范围内（正值）
        daily_return = np.random.normal(self.drift, self.volatility) # 生成随机价格变化（服从正态分布）
        self.price *= (1 + daily_return) # 更新当前股票价格（根据趋势和波动率） 
        self.price = max(self.price, 0.1) # 确保价格不低于0.1元（防止价格异常低）   
        
        # 3. 推进时间
        self.day += 1
        
        # 4. 可能更新新闻
        if random.random() < 0.1: # 10%的概率有新新闻
            self.market_news = random.choice(["预计将有积极的收益报告。", "新竞争对手进入市场。", "宏观经济前景强劲。", "监管担忧日益增长。"])
            # 使用LLM分析新闻影响
            self._analyze_news_impact()
        else:
             self.drift = 0.01 # 恢复正常趋势

    def get_state_string(self) -> str:
        return f"第 {self.day} 天: 价格=${self.price:.2f}, 新闻: {self.market_news}\n投资组合: ${self.portfolio.value(self.price):.2f} ({self.portfolio.shares} 股, ${self.portfolio.cash:.2f} 现金)"

print("市场模拟器环境定义成功。")

# --- 构建模拟器智能体 ---

# Pydantic模型用于LLM的输出，确保步骤之间的结构化通信

class ProposedAction(BaseModel):
    """分析师提出的高级策略。""" 
    strategy: str = Field(description="高级交易策略，例如，'激进买入'，'谨慎卖出'，'观望'")
    reasoning: str = Field(description="提出该策略的简要理由")

class FinalDecision(BaseModel):
    """最终的、具体的执行动作。"""
    action: str = Field(description="要采取的最终行动：'buy'，'sell'，或'hold'")
    amount: float = Field(description="要买入或卖出的股票数量。如果持有，则应为0")
    reasoning: str = Field(description="最终决策的理由，参考模拟结果")

# LangGraph状态，包含真实市场、分析师策略、模拟结果和最终决策
class AgentState(TypedDict):# TypedDict定义了智能体状态的结构，包含真实市场、分析师策略、模拟结果和最终决策
    real_market: MarketSimulator # 真实市场模拟器实例，用于执行交易和更新市场状态
    proposed_action: Optional[ProposedAction] # 分析师提出的交易策略（可选）
    simulation_results: Optional[List[Dict]] # 多个模拟场景的结果（可选）
    final_decision: Optional[FinalDecision] # 最终的交易决策（可选）    

# 初始化LLM
llm = ModelScopeChat(temperature=0.4)

# 图节点

def propose_action_node(state: AgentState) -> Dict[str, Any]:
    """观察市场并提出高级策略。
    
    Args:
        state: 当前智能体状态，包含真实市场信息
    
    Returns:
        更新后的状态，包含分析师提出的交易策略
    """
    console.print("--- 🧐 分析师提出策略 ---")
    # 构建提示，要求LLM基于当前市场状态提出交易策略
    market_state = state['real_market'].get_state_string()
    prompt = f"你是一位敏锐的金融分析师。根据当前市场状态，提出一个交易策略。\n\n市场状态:\n{market_state}"
    # 设置LLM输出结构为ProposedAction模型
    proposer_llm = llm.with_structured_output(ProposedAction)
    # 直接调用LLM获取策略提议
    proposal = proposer_llm.invoke(prompt)
    # 打印提议结果
    console.print(f"[yellow]提议:[/yellow] {proposal.strategy}. [italic]理由: {proposal.reasoning}[/italic]")
    # 返回包含提议策略的更新状态
    return {"proposed_action": proposal}

def run_simulation_node(state: AgentState) -> Dict[str, Any]:
    """在沙盒模拟中运行提议的策略。
    
    Args:
        state: 当前智能体状态，包含分析师提出的策略
    
    Returns:
        更新后的状态，包含多个模拟场景的结果
    """
    console.print("--- 🤖 运行模拟 ---")
    
    # 获取分析师提出的策略
    strategy = state['proposed_action'].strategy
    
    # 模拟参数配置
    num_simulations = 5  # 运行5个不同的模拟场景
    simulation_horizon = 10  # 每个模拟持续10天
    results = []  # 存储模拟结果

    # 运行多个模拟场景
    for i in range(num_simulations):
        # 重要：创建市场状态的深拷贝，确保模拟不会影响真实市场
        simulated_market = state['real_market'].model_copy(deep=True)
        
        # 记录初始投资组合价值
        initial_value = simulated_market.portfolio.value(simulated_market.price)

        # 将高级策略转换为具体的交易行动
        if "买入" in strategy:
            action = "buy"
            # 根据策略的激进程度决定投入资金比例
            amount = (simulated_market.portfolio.cash * (0.25 if "激进" in strategy else 0.1)) / simulated_market.price
        elif "卖出" in strategy:
            action = "sell"
            # 根据策略的激进程度决定卖出股票比例
            amount = simulated_market.portfolio.shares * (0.25 if "激进" in strategy else 0.1)
        else:
            action = "hold"
            amount = 0
        
        # 执行初始交易
        simulated_market.step(action, amount)
        
        # 继续模拟后续天数（只持有，不执行新交易）
        for _ in range(simulation_horizon - 1):
            simulated_market.step("hold")
        
        # 记录最终投资组合价值和回报
        final_value = simulated_market.portfolio.value(simulated_market.price)
        results.append({
            "sim_num": i+1, # 模拟场景编号
            "initial_value": initial_value, # 初始投资组合价值，单位：美元
            "final_value": final_value, # 最终投资组合价值，单位：美元  
            "return_pct": (final_value - initial_value) / initial_value * 100 # 回报百分比  
        })
    
    console.print("[cyan]模拟完成。结果将传递给风险经理。[/cyan]")
    
    # 返回包含模拟结果的更新状态
    return {"simulation_results": results}

def refine_and_decide_node(state: AgentState) -> Dict[str, Any]:
    """分析模拟结果并做出最终的、完善的决策。"""
    console.print("--- 🧠 风险经理完善决策 ---")
    results_summary = "\n".join([f"模拟 {r['sim_num']}: 初始=${r['initial_value']:.2f}, 最终=${r['final_value']:.2f}, 回报={r['return_pct']:.2f}%" for r in state['simulation_results']])
    
    # 构建提示
    prompt = f"你是一位谨慎的风险经理。你的分析师提出了一个策略。你已经运行了模拟来测试它。根据潜在结果，做出最终的、具体的决策。如果结果高度可变或为负，降低风险（例如，少买/卖一些股票，或持有）。\n\n初始提议: {state['proposed_action'].strategy}\n\n模拟结果:\n{results_summary}\n\n真实市场状态:\n{state['real_market'].get_state_string()}"
    
    # 设置LLM输出结构为FinalDecision模型
    decider_llm = llm.with_structured_output(FinalDecision)
    # 直接调用LLM获取最终决策
    final_decision = decider_llm.invoke(prompt)
    
    console.print(f"[green]最终决策:[/green] {final_decision.action} {final_decision.amount:.0f} 股. [italic]理由: {final_decision.reasoning}[/italic]")
    return {"final_decision": final_decision}

def execute_in_real_world_node(state: AgentState) -> Dict[str, Any]:
    """在真实市场环境中执行最终决策。"""
    console.print("--- 🚀 在真实世界执行 ---")
    decision = state['final_decision']
    real_market = state['real_market']
    real_market.step(decision.action, decision.amount)
    console.print(f"[bold]执行完成。新市场状态:[/bold]\n{real_market.get_state_string()}")
    return {"real_market": real_market}

# 构建图
workflow = StateGraph(AgentState)
workflow.add_node("propose", propose_action_node)
workflow.add_node("simulate", run_simulation_node)
workflow.add_node("refine", refine_and_decide_node)
workflow.add_node("execute", execute_in_real_world_node)

workflow.set_entry_point("propose")
workflow.add_edge("propose", "simulate")
workflow.add_edge("simulate", "refine")
workflow.add_edge("refine", "execute")
workflow.add_edge("execute", END)

simulator_agent = workflow.compile()
print("循环思维智能体图编译成功。")

# --- 演示 ---

def run_demo():
    """运行演示，展示智能体如何在市场中行动几天。"""
    console.print("--- 初始市场状态 ---")
    # 创建初始市场状态
    market = MarketSimulator()
    console.print(market.get_state_string())
    
    console.print("\n--- 第 1 天: 好消息传来! ---")
    market.market_news = "预计将有积极的收益报告。"
    
    # 运行智能体
    result = simulator_agent.invoke({
        "real_market": market,
        "proposed_action": None,
        "simulation_results": None,
        "final_decision": None
    })
    
    market = result["real_market"]
    
    console.print("\n--- 第 2 天: 坏消息传来! ---")
    market.market_news = "新竞争对手进入市场。"
    
    # 再次运行智能体
    result = simulator_agent.invoke({
        "real_market": market,
        "proposed_action": None,
        "simulation_results": None,
        "final_decision": None
    })
    
    market = result["real_market"]
    
    console.print("\n--- 第 3 天: 市场稳定 ---")
    market.market_news = "市场稳定。"
    
    # 再次运行智能体
    result = simulator_agent.invoke({
        "real_market": market,
        "proposed_action": None,
        "simulation_results": None,
        "final_decision": None
    })

# --- 主函数 ---

if __name__ == "__main__":
    console.print("🚀 启动思维模型循环智能体...")
    console.print("📚 这是一个股票交易智能体，演示了思维模型循环架构")
    console.print("💡 智能体将在模拟市场中运行3天，处理好消息和坏消息")
    console.print("\n按 Enter 键开始演示...")
    input()
    
    # 检查是否设置了API密钥
    if not os.environ.get("MODELSCOPE_API_KEY"):
        console.print("[bold yellow]⚠️ 未设置MODELSCOPE_API_KEY环境变量")
        console.print("将使用模拟响应进行演示。要使用真实的LLM，请在.env文件中设置API密钥。[/bold yellow]")
        input("按 Enter 键继续...")
    
    run_demo()
    
    console.print("\n🎉 演示完成!")
    console.print("📊 思维模型循环架构成功地让智能体在执行前模拟了其行动")
    console.print("💪 这种方法提高了安全性，并减少了高风险环境中的错误")

# -*- coding: utf-8 -*-
"""
工具使用（Tool Use）架构的可运行示例
 
学习目标（看完你能做到什么）：
- 了解“工具使用”智能体如何把复杂任务拆解为“规划→执行→汇总”
- 理解 LangGraph 如何把多步逻辑编排成“有状态的工作流”
- 学会用 Pydantic v2 约束 LLM 输出为结构化数据（更稳、更好用）
- 能把脚本作为命令行程序运行，并观察每一步的结构化中间结果
 
核心概念速览：
- StateGraph：有状态的“流程图”，每个节点是一个函数，节点之间按边连接顺序执行
- 工具注册表（TOOLS_REGISTRY）：把可用的本地工具统一声明，便于 LLM 进行“规划”与“调用”
- 三步节点：规划（Planner）→ 执行（Executor）→ 汇总（Summarizer）
 
运行前准备：
- 项目根目录创建 `.env` 并配置：
  - `LANGCHAIN_API_KEY`（用于 LangSmith 追踪，可选）
  - 如使用 ModelScope 接入：`MODELSCOPE_BASE_URL`、`MODELSCOPE_API_KEY`、`MODELSCOPE_MODEL_ID`
  - 如使用高德MCP服务：
    - `AMAP_KEY`：在高德开放平台申请的API密钥（https://console.amap.com/）
    - `AMAP_MCP_URL`：高德MCP服务器地址（可选，默认：https://mcp.amap.com/mcp）
 
如何运行：
- 直接运行默认示例：`python 02_tool_use.py`
- 更换任务请求：`python 02_tool_use.py --request "请计算表达式 2+3*4，并列出当前目录文件。"`
- 开启详细教学日志：`python 02_tool_use.py --debug`
 
阅读建议：
- 先从“数据模型”和“三个节点函数”开始理解，再看“build_app”和“run_workflow”如何把所有部件串起来
- 本示例的工具为安全本地函数，便于学习“工具调用”与“结构化编排”的基本套路
"""
import os
import json
import argparse
import sys
import platform
from typing import List, TypedDict, Optional, Dict, Any
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, END
from rich.console import Console
from rich.syntax import Syntax
from openai import OpenAI, RateLimitError, APIError
import logging
from rich.logging import RichHandler

console = Console()
DEBUG: bool = True
STREAM_TOKENS: bool = False
MAX_STEPS: int = 10
ON_ERROR: str = "skip"
logger = logging.getLogger("tool_use")
handler = RichHandler(console=console, rich_tracebacks=True, markup=True)
handler.setFormatter(logging.Formatter("%(message)s"))
logger.handlers = [handler]
logger.propagate = False
logger.setLevel(logging.DEBUG if DEBUG else logging.INFO)

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
    def invoke(self, prompt: str, stream_tokens: bool = False):
        try:
            if stream_tokens:
                resp_iter = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=self.temperature,
                    stream=True,
                    extra_body=self.extra_body,
                )
                buffer = []
                import sys as _sys
                for chunk in resp_iter:
                    delta = getattr(chunk.choices[0], "delta", None)
                    token = getattr(delta, "content", "") if delta else ""
                    if token:
                        buffer.append(token)
                        _sys.stdout.write(token)
                        _sys.stdout.flush()
                return "".join(buffer)
            else:
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
                if DEBUG:
                    console.print(f"[bold yellow]⚠️ 主模型请求失败：{e}，尝试切换到备用模型[/bold yellow]")
                self.model = self.fallback_model
                self.switched = True
                # 重试请求
                if stream_tokens:
                    resp_iter = self.client.chat.completions.create(
                        model=self.model,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=self.temperature,
                        stream=True,
                        extra_body=self.extra_body,
                    )
                    buffer = []
                    import sys as _sys
                    for chunk in resp_iter:
                        delta = getattr(chunk.choices[0], "delta", None)
                        token = getattr(delta, "content", "") if delta else ""
                        if token:
                            buffer.append(token)
                            _sys.stdout.write(token)
                            _sys.stdout.flush()
                    return "".join(buffer)
                else:
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
                # 通过系统提示约束仅输出 JSON（尽量提高解析成功率），并明确字段/类型
                schema = pyd_model.model_json_schema()
                props = schema.get("properties", {})
                required = schema.get("required", [])
                schema_text_lines = []
                for k, v in props.items():
                    t = v.get("type", "string")
                    schema_text_lines.append(f"- {k}: {t}")
                schema_text = "\n".join(schema_text_lines) or "- fields"
                required_text = ", ".join(required) if required else "all"
                system_msg = (
                    "只输出一个 JSON 对象，严格匹配以下字段与类型：\n"
                    f"{schema_text}\n"
                    f"必须包含字段：{required_text}\n"
                    "不要输出任何解释或多余文本。"
                )
                messages = [
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": prompt},
                ]
                try:
                    if STREAM_TOKENS:
                        content_iter = self.outer.client.chat.completions.create(
                            model=self.outer.model,
                            messages=messages,
                            temperature=self.outer.temperature,
                            stream=True,
                            extra_body=self.outer.extra_body,
                        )
                        import sys as _sys
                        _sys.stdout.write("\n📡 正在接收结构化 JSON...\n")
                        _sys.stdout.flush()
                        parts = []
                        for chunk in content_iter:
                            delta = getattr(chunk.choices[0], "delta", None)
                            token = getattr(delta, "content", "") if delta else ""
                            if token:
                                parts.append(token)
                                _sys.stdout.write(token)
                                _sys.stdout.flush()
                        content = "".join(parts)
                    else:
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
                        if DEBUG:
                            console.print(f"[bold yellow]⚠️ 主模型请求失败：{e}，尝试切换到备用模型[/bold yellow]")
                        self.outer.model = self.outer.fallback_model
                        self.outer.switched = True
                        # 重试请求
                        if STREAM_TOKENS:
                            content_iter = self.outer.client.chat.completions.create(
                                model=self.outer.model,
                                messages=messages,
                                temperature=self.outer.temperature,
                                stream=True,
                                extra_body=self.outer.extra_body,
                            )
                            import sys as _sys
                            _sys.stdout.write("\n📡 正在接收结构化 JSON...\n")
                            _sys.stdout.flush()
                            parts = []
                            for chunk in content_iter:
                                delta = getattr(chunk.choices[0], "delta", None)
                                token = getattr(delta, "content", "") if delta else ""
                                if token:
                                    parts.append(token)
                                    _sys.stdout.write(token)
                                    _sys.stdout.flush()
                            content = "".join(parts)
                        else:
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
                import json as _json, re
                from pydantic import ValidationError
                def _extract_json(s: str) -> str:
                    m = re.search(r'(\{[\s\S]*\}|\[[\s\S]*\])', s)
                    return m.group(1) if m else "{}"
                raw = content.strip()
                try:
                    data = _json.loads(raw)
                except Exception:
                    data = _json.loads(_extract_json(raw))
                try:
                    parsed = pyd_model.model_validate(data)
                    return parsed
                except ValidationError:
                    mappings_applied = False
                    # 兜底字段映射：尽量把常见别名映射到目标模型字段
                    if pyd_model.__name__ == "ToolPlan":
                        if "planned_calls" not in data and "tools" in data:
                            data["planned_calls"] = data.pop("tools")
                            mappings_applied = True
                        if "planned_calls" in data and isinstance(data["planned_calls"], list):
                            normalized = []
                            for item in data["planned_calls"]:
                                if not isinstance(item, dict):
                                    continue
                                tname = item.get("tool_name") or item.get("name") or item.get("tool")
                                args = item.get("arguments") or item.get("args") or item.get("parameters") or {}
                                reason = item.get("reason") or item.get("why") or "auto"
                                assign_to = item.get("assign_to") or item.get("assign") or item.get("save_as") or item.get("var") or None
                                normalized.append({"tool_name": tname, "arguments": args, "reason": reason, "assign_to": assign_to})
                            data["planned_calls"] = normalized
                            mappings_applied = True
                    if pyd_model.__name__ == "FinalAnswer":
                        if "answer" not in data and "final" in data:
                            data["answer"] = data.pop("final")
                            mappings_applied = True
                        if "sources" not in data and "refs" in data:
                            data["sources"] = data.pop("refs")
                            mappings_applied = True
                    parsed = pyd_model.model_validate(data if mappings_applied else data)
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
    - 额外参数：enable_thinking 可选；强制 JSON 输出
    - 当主模型请求失败时，会自动切换到备用模型（如果配置了的话）
    """
    base_url = os.environ.get("MODELSCOPE_BASE_URL", "https://api-inference.modelscope.cn/v1")
    api_key = os.environ.get("MODELSCOPE_API_KEY", "")
    model_id = os.environ.get("MODELSCOPE_MODEL_ID", "deepseek-ai/DeepSeek-V3.2")
    fallback_model_id = os.environ.get("MODELSCOPE_MODEL_ID_R1")
    extra = {
        "enable_thinking": True,
        "trust_request_chat_template": True,
        "response_format": {"type": "json_object"},
    }
    return ModelScopeChat(base_url=base_url, api_key=api_key, model=model_id, fallback_model=fallback_model_id, temperature=0.2, extra_body=extra)

class ToolCall(BaseModel):
    """单次工具调用的计划项：选用工具、传入参数、调用理由"""
    tool_name: str = Field(description="工具名称")
    arguments: Dict[str, Any] = Field(description="工具入参对象")
    reason: str = Field(description="调用该工具的原因")
    assign_to: Optional[str] = Field(default=None, description="将该工具输出保存到上下文中的变量名，供后续步骤引用")

class ToolPlan(BaseModel):
    """整体工具调用计划：按顺序排列多个 ToolCall，并给出计划摘要"""
    planned_calls: List[ToolCall] = Field(description="按顺序计划调用的工具列表")
    plan_summary: str = Field(description="整体计划摘要")

class ToolResult(BaseModel):
    """单次工具调用的结构化结果：记录工具名与输出"""
    tool_name: str = Field(description="工具名称")
    output: Any = Field(description="工具输出")

class ToolExecutionResults(BaseModel):
    """执行多个工具调用后的汇总结果：结果列表与执行过程摘要"""
    results: List[ToolResult] = Field(description="工具执行结果列表")
    execution_summary: str = Field(description="执行过程摘要")
    context: Dict[str, Any] = Field(default_factory=dict, description="累计的上下文变量字典（含每步 assign_to 的输出）")

class FinalAnswer(BaseModel):
    """最终回答：综合工具结果后的答案与引用来源"""
    answer: str = Field(description="综合工具结果后的最终回答")
    sources: List[str] = Field(description="引用的工具或来源标识")

class ToolUseState(TypedDict):
    """工作流状态：在节点间传递的共享数据"""
    user_request: str
    tool_plan: Optional[dict]
    tool_results: Optional[dict]
    final_answer: Optional[dict]

def _safe_eval(expr: str) -> float:
    """
    安全算术表达式求值：
    - 使用 AST 限制可用操作符，避免任意代码执行风险
    - 支持 +, -, *, /, ** 以及一元负号
    """
    import ast, operator as op
    allowed = {
        ast.Add: op.add,
        ast.Sub: op.sub,
        ast.Mult: op.mul,
        ast.Div: op.truediv,
        ast.Pow: op.pow,
        ast.USub: op.neg,
    }
    def _eval(node):
        if isinstance(node, ast.Num):
            return node.n
        if isinstance(node, ast.BinOp) and type(node.op) in allowed:
            return allowed[type(node.op)](_eval(node.left), _eval(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in allowed:
            return allowed[type(node.op)](_eval(node.operand))
        raise ValueError("unsupported expression")
    tree = ast.parse(expr, mode="eval")
    return float(_eval(tree.body))

def tool_calc(arguments: Dict[str, Any]) -> Dict[str, Any]:
    """工具：计算算术表达式的值（使用 _safe_eval 保证安全）"""
    expr = str(arguments.get("expression", ""))
    value = _safe_eval(expr)
    return {"expression": expr, "value": value}

def tool_python_info(arguments: Dict[str, Any]) -> Dict[str, Any]:
    """工具：返回当前 Python 版本、实现与平台信息"""
    return {
        "version": sys.version.split()[0],
        "implementation": platform.python_implementation(),
        "platform": platform.platform(),
    }

def tool_string_length(arguments: Dict[str, Any]) -> Dict[str, Any]:
    """工具：返回字符串长度与字符统计，纯内存无权限风险"""
    s = str(arguments.get("text", ""))
    return {"text": s, "length": len(s)}

def tool_concat(arguments: Dict[str, Any]) -> Dict[str, Any]:
    """工具：拼接两个字符串为一个结果"""
    a = str(arguments.get("a", ""))
    b = str(arguments.get("b", ""))
    return {"result": a + b}

def tool_repeat(arguments: Dict[str, Any]) -> Dict[str, Any]:
    """工具：将字符串重复指定次数"""
    text = str(arguments.get("text", ""))
    times = int(arguments.get("times", 1))
    times = max(0, min(times, 1000))
    return {"result": text * times}

def tool_upper(arguments: Dict[str, Any]) -> Dict[str, Any]:
    """工具：将字符串转为大写"""
    text = str(arguments.get("text", ""))
    return {"result": text.upper()}

def tool_normalize_text(arguments: Dict[str, Any]) -> Dict[str, Any]:
    """工具：标准化文本（小写、去除常见标点、压缩空白）"""
    import re
    text = str(arguments.get("text", ""))
    lower = text.lower()
    no_punct = re.sub(r"[^\w\s]", " ", lower)
    normalized = re.sub(r"\s+", " ", no_punct).strip()
    return {"normalized": normalized, "orig": text}

def tool_tokenize_words(arguments: Dict[str, Any]) -> Dict[str, Any]:
    """工具：按空白分词，返回词列表"""
    text = str(arguments.get("text", ""))
    tokens = [t for t in text.split() if t]
    return {"tokens": tokens}

def tool_remove_stopwords(arguments: Dict[str, Any]) -> Dict[str, Any]:
    """工具：移除常见英文停用词（示例用，小集合），保持中文词不变"""
    tokens = list(arguments.get("tokens", []))
    stop = {"the","a","an","and","or","of","to","in","on","for","with","is","are","be","this","that","it"}
    cleaned = [t for t in tokens if t.lower() not in stop]
    return {"tokens": cleaned}

def tool_keyword_extract(arguments: Dict[str, Any]) -> Dict[str, Any]:
    """工具：基于词频的简易关键词提取，返回 top_k 关键词与计数"""
    tokens = list(arguments.get("tokens", []))
    top_k = int(arguments.get("top_k", 5))
    from collections import Counter
    cnt = Counter(tokens)
    top = cnt.most_common(max(1, top_k))
    return {"keywords": [{"term": t, "count": c} for t, c in top]}

def tool_word_count(arguments: Dict[str, Any]) -> Dict[str, Any]:
    """工具：统计字符串的单词数量（以空白分隔），纯内存安全"""
    s = str(arguments.get("text", ""))
    words = [w for w in s.split() if w]
    return {"text": s, "word_count": len(words)}

def tool_current_time(arguments: Dict[str, Any]) -> Dict[str, Any]:
    """工具：返回当前本地时间字符串，不涉及外部访问"""
    from datetime import datetime
    fmt = str(arguments.get("format", "%Y-%m-%d %H:%M:%S"))
    return {"now": datetime.now().strftime(fmt), "format": fmt}

def tool_title_case(arguments: Dict[str, Any]) -> Dict[str, Any]:
    """工具：标题化文本（英文示例，中文基本保持不变）"""
    text = str(arguments.get("text", ""))
    return {"title": text.title()}

def tool_slugify(arguments: Dict[str, Any]) -> Dict[str, Any]:
    """工具：生成 URL 友好的 slug（简化规则）"""
    import re
    text = str(arguments.get("text", ""))
    lower = text.lower().strip()
    slug = re.sub(r"[^\w\s-]", "", lower)
    slug = re.sub(r"\s+", "-", slug)
    slug = re.sub(r"-{2,}", "-", slug).strip("-")
    return {"slug": slug}

def tool_render_report(arguments: Dict[str, Any]) -> Dict[str, Any]:
    """工具：将管线各步结果渲染为 Markdown 报告，便于展示学习成果"""
    original = str(arguments.get("original", ""))
    normalized = str(arguments.get("normalized", ""))
    keywords = list(arguments.get("keywords", []))
    sentiment = dict(arguments.get("sentiment", {}))
    now = str(arguments.get("time", ""))
    import re
    # 自动生成标题与 slug（若未提供）
    title = str(arguments.get("title", "")).strip()
    slug = str(arguments.get("slug", "")).strip()
    if not title:
        # 取标准化文本的前若干词作为标题
        head = " ".join(normalized.split()[:8]) if normalized else "Processed Text Report"
        title = head.title()
    if not slug:
        lower = normalized.lower().strip()
        slug = re.sub(r"[^\w\s-]", "", lower)
        slug = re.sub(r"\s+", "-", slug)
        slug = re.sub(r"-{2,}", "-", slug).strip("-") or "processed-text-report"
    lines = []
    lines.append(f"# {title}")
    lines.append(f"- Time: {now}")
    lines.append(f"- Slug: `{slug}`")
    lines.append("")
    lines.append("## Original")
    lines.append(f"{original}")
    lines.append("")
    lines.append("## Normalized")
    lines.append(f"{normalized}")
    lines.append("")
    lines.append("## Keywords")
    if keywords:
        lines.append("\n".join([f"- {k.get('term')} ({k.get('count')})" for k in keywords]))
    else:
        lines.append("- (none)")
    lines.append("")
    lines.append("## Sentiment")
    if sentiment:
        lines.append(f"- Label: {sentiment.get('label')}")
        lines.append(f"- Score: {sentiment.get('score')}")
    else:
        lines.append("- (n/a)")
    return {"markdown": "\n".join(lines)}

def tool_amap_mcp(arguments: Dict[str, Any]) -> Dict[str, Any]:
    """工具：调用高德MCP服务，获取地图相关数据
    
    高德MCP服务器支持的主要功能包括：
    - 地理编码（地址转坐标）
    - 逆地理编码（坐标转地址）
    - 路线规划
    - 兴趣点查询
    - 地图数据查询等
    
    参数示例：
    {"service": "geocode", "parameters": {"address": "北京市朝阳区"}}
    {"service": "regeo", "parameters": {"location": "116.407413,39.904211"}}
    
    返回示例：
    {"status": "success", "data": {"geocodes": [...]}, "url": "请求URL"}
    """
    import os, requests
    
    # 从环境变量获取配置
    amap_mcp_url = os.environ.get("AMAP_MCP_URL", "https://mcp.amap.com/mcp")
    amap_key = os.environ.get("AMAP_KEY", "")
    
    if not amap_key:
        return {"error": "AMAP_KEY 未在环境变量中配置，请前往 https://console.amap.com/ 申请"}
    
    # 解析MCP服务参数
    service = arguments.get("service")
    parameters = arguments.get("parameters", {})
    
    if not service:
        return {"error": "缺少必要参数 'service'，请指定要调用的高德MCP服务"}
    
    # 构建请求参数：将API key和服务参数合并
    params = {"key": amap_key, "service": service, **parameters}
    
    try:
        # 发送请求到高德MCP服务器（使用POST方法，高德MCP服务通常要求POST请求）
        response = requests.post(amap_mcp_url, json=params, timeout=15)
        response.raise_for_status()  # 检查HTTP错误
        
        # 解析响应
        result = response.json()
        
        # 检查高德API返回的状态码
        if result.get("status") != "1":
            error_info = result.get("info", "未知错误")
            error_code = result.get("infocode", "0")
            return {"status": "error", "error": f"高德API错误: {error_info} (错误码: {error_code})", "url": response.url}
        
        return {"status": "success", "data": result, "url": response.url}
    except requests.exceptions.RequestException as e:
        return {"status": "error", "error": f"请求失败: {str(e)}", "url": amap_mcp_url}
    except ValueError as e:
        return {"status": "error", "error": f"响应解析失败: {str(e)}", "url": f"{amap_mcp_url}?{'&'.join([f'{k}={v}' for k, v in params.items()])}"}

def tool_sentiment_simple(arguments: Dict[str, Any]) -> Dict[str, Any]:
    """工具：基于简易词典的情感判断（英文示例）"""
    text = str(arguments.get("text", ""))
    pos = {"good","great","excellent","love","happy","awesome","nice","cool","fast","smart"}
    neg = {"bad","terrible","awful","hate","sad","slow","stupid","bug","issue"}
    import re
    tokens = re.findall(r"\w+", text.lower())
    p = sum(1 for t in tokens if t in pos)
    n = sum(1 for t in tokens if t in neg)
    score = float(p - n)
    label = "positive" if score > 0 else ("negative" if score < 0 else "neutral")
    return {"score": score, "label": label}

TOOLS_REGISTRY = {
    # 工具注册表：统一声明可用的本地工具，便于“规划节点”引用
    "normalize_text": tool_normalize_text,
    "tokenize_words": tool_tokenize_words,
    "keyword_extract": tool_keyword_extract,
    "current_time": tool_current_time,
    "render_report": tool_render_report,
    "amap_mcp": tool_amap_mcp,  # 高德MCP地图服务工具（基于Model Context Protocol）
}

def make_planner_node(llm: "ModelScopeChat"):
    """规划节点：让 LLM 选择并编排可用工具调用，输出结构化计划（ToolPlan）"""
    planner_llm = llm.with_structured_output(ToolPlan)
    def _node(state: ToolUseState) -> dict:
        if DEBUG:
            console.print("[bold]规划工具调用[/bold]")
        tool_catalog = json.dumps(
            {
                "available_tools": [
                    {"name": "normalize_text", "args": {"text": "原始文本"}},
                    {"name": "tokenize_words", "args": {"text": "要分词的文本"}},
                    {"name": "keyword_extract", "args": {"tokens": "词列表", "top_k": "关键词数量（整数）"}},
                    {"name": "current_time", "args": {"format": "可选，时间格式，默认 '%Y-%m-%d %H:%M:%S'"}},
                    {"name": "render_report", "args": {"original": "原文", "normalized": "标准化文本", "keywords": "关键词列表", "time": "时间字符串"}},
                    {"name": "amap_mcp", "args": {"service": "高德MCP服务名称（如：geocode, regeo, route, poi）", "parameters": "服务参数对象（根据不同服务类型）"}},  # 高德MCP地图服务（基于Model Context Protocol）
                ]
            },
            ensure_ascii=False,
            indent=2,
        )
        prompt = (
            "你是工具规划助手。根据用户请求选择并编排可用工具调用，返回 JSON 计划。\n"
            "要求：\n"
            "1) 每个步骤可使用 assign_to 将输出保存为变量名，以便后续步骤通过 $变量名 或 $变量名.字段 引用。\n"
            "2) 初始上下文包含变量：$request（即用户请求文本）。\n"
            "3) 仅使用给定的可用工具，必要时多步串联，确保后续步骤参数可从前序变量中取值。\n\n"
            "示例（文本处理管线，精简版）：\n"
            "{\n"
            "  \"planned_calls\": [\n"
            "    {\"tool_name\": \"normalize_text\", \"arguments\": {\"text\": \"$request\"}, \"reason\": \"标准化原文\", \"assign_to\": \"normalized\"},\n"
            "    {\"tool_name\": \"tokenize_words\", \"arguments\": {\"text\": \"$normalized.normalized\"}, \"reason\": \"分词\", \"assign_to\": \"tokens\"},\n"
            "    {\"tool_name\": \"keyword_extract\", \"arguments\": {\"tokens\": \"$tokens.tokens\", \"top_k\": 5}, \"reason\": \"提取关键词\", \"assign_to\": \"keywords\"},\n"
            "    {\"tool_name\": \"current_time\", \"arguments\": {\"format\": \"%Y-%m-%d %H:%M\"}, \"reason\": \"记录时间\", \"assign_to\": \"time\"},\n"
            "    {\"tool_name\": \"render_report\", \"arguments\": {\"original\": \"$normalized.orig\", \"normalized\": \"$normalized.normalized\", \"keywords\": \"$keywords.keywords\", \"time\": \"$time.now\"}, \"reason\": \"生成最终报告\", \"assign_to\": \"report\"}\n"
            "  ],\n"
            "  \"plan_summary\": \"标准化→分词→关键词→时间→报告\"\n"
            "}\n\n"
            f"工具目录：\n{tool_catalog}\n\n"
            f"用户请求：\n{state['user_request']}\n"
        )
        from rich.console import Console as _Console
        _status_console = _Console()
        with _status_console.status("规划中...", spinner="dots"):
            plan = planner_llm.invoke(prompt)
        return {"tool_plan": plan.model_dump()}
    return _node

def make_plan_check_node():
    def _node(state: ToolUseState) -> dict:
        plan = state.get("tool_plan") or {}
        calls = list(plan.get("planned_calls", []))
        warnings = []
        # 步数限制
        if len(calls) > MAX_STEPS:
            warnings.append(f"planned_calls exceeded MAX_STEPS={MAX_STEPS}, truncating")
            calls = calls[:MAX_STEPS]
        # 工具与变量校验、assign_to 兜底与去重
        seen_vars = {"request"}
        used_names = set()
        cleaned = []
        for idx, item in enumerate(calls):
            name = item.get("tool_name")
            args = item.get("arguments", {}) or {}
            assign_to = item.get("assign_to") or f"{name}_{idx}"
            if not name or name not in TOOLS_REGISTRY:
                warnings.append(f"unknown tool '{name}' at step {idx}, skipping")
                continue
            base_assign = assign_to
            suffix = 1
            while assign_to in used_names:
                assign_to = f"{base_assign}_{suffix}"
                suffix += 1
            used_names.add(assign_to)
            # 变量引用基础校验（只校验基变量是否已定义）
            def _base_vars(v):
                if isinstance(v, str) and v.startswith("$"):
                    return v[1:].split(".")[0]
                return None
            missing_refs = []
            for v in args.values():
                b = _base_vars(v)
                if b and b not in seen_vars:
                    missing_refs.append(b)
            if missing_refs:
                warnings.append(f"step {idx} references undefined vars: {sorted(set(missing_refs))}, keeping step but may fail")
            cleaned.append({"tool_name": name, "arguments": args, "reason": item.get("reason", "auto"), "assign_to": assign_to})
            seen_vars.add(assign_to)
        plan_checked = {
            "planned_calls": cleaned,
            "plan_summary": (plan.get("plan_summary") or "") + ((" | warnings: " + "; ".join(warnings)) if warnings else "")
        }
        if DEBUG and warnings:
            console.print("[bold yellow]规划检查警告[/bold yellow]: " + "; ".join(warnings))
        return {"tool_plan": plan_checked}
    return _node

def make_executor_node():
    """执行节点：依次执行工具计划，返回结构化的执行结果（ToolExecutionResults）"""
    def _resolve_value(value: Any, ctx: Dict[str, Any]) -> Any:
        # 字符串引用解析："$var" 或 "$var.field"
        if isinstance(value, str) and value.startswith("$"):
            path = value[1:]
            parts = path.split(".")
            base = ctx.get(parts[0], None)
            if base is None:
                return None
            cur = base
            for p in parts[1:]:
                if isinstance(cur, dict):
                    cur = cur.get(p, None)
                else:
                    cur = getattr(cur, p, None) if hasattr(cur, p) else None
            return cur
        return value
    def _resolve_args(args: Dict[str, Any], ctx: Dict[str, Any]) -> Dict[str, Any]:
        def _resolve(obj: Any) -> Any:
            if isinstance(obj, dict):
                return {k: _resolve(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [_resolve(v) for v in obj]
            return _resolve_value(obj, ctx)
        return _resolve(args or {})
    def _node(state: ToolUseState) -> dict:
        if DEBUG:
            console.print("[bold]执行工具计划[/bold]")
        planned = state["tool_plan"]["planned_calls"] if state.get("tool_plan") else []
        results: List[Dict[str, Any]] = []
        context: Dict[str, Any] = {"request": state["user_request"]}
        errors = 0
        for call in planned:
            name = call.get("tool_name")
            args = call.get("arguments", {}) or {}
            assign_to = call.get("assign_to") or f"{name}"
            func = TOOLS_REGISTRY.get(name)
            if func is None:
                results.append({"tool_name": name, "output": {"error": "unknown tool"}})
                errors += 1
                continue
            try:
                resolved_args = _resolve_args(args, context)
                out = func(resolved_args)
                results.append({"tool_name": name, "output": out})
                context[assign_to] = out
            except Exception as e:
                results.append({"tool_name": name, "output": {"error": str(e)}})
                errors += 1
                if ON_ERROR == "stop":
                    break
                elif ON_ERROR == "skip":
                    continue
                elif ON_ERROR == "fallback":
                    continue
        summary = "done" + (f" with {errors} error(s), policy={ON_ERROR}" if errors else "")
        payload = ToolExecutionResults(
            results=[ToolResult(tool_name=r["tool_name"], output=r["output"]) for r in results],
            execution_summary=summary,
            context=context,
        )
        return {"tool_results": payload.model_dump()}
    return _node

def make_summarizer_node(llm: "ModelScopeChat"):
    """汇总节点：综合工具执行结果，生成最终回答与引用来源（FinalAnswer）"""
    summarizer_llm = llm.with_structured_output(FinalAnswer)
    def _node(state: ToolUseState) -> dict:
        if DEBUG:
            console.print("[bold]汇总生成答案[/bold]")
        req = state["user_request"]
        tool_results_json = json.dumps(state.get("tool_results", {}), ensure_ascii=False, indent=2)
        context_json = json.dumps(state.get("tool_results", {}).get("context", {}), ensure_ascii=False, indent=2)
        prompt = (
            "你是结果汇总器。根据用户请求与工具执行结果，生成最终回答与引用来源列表。\n"
            "若工具结果含错误，需说明并给出可行的替代建议。\n\n"
            f"用户请求：\n{req}\n\n"
            f"工具结果：\n{tool_results_json}\n\n"
            f"上下文变量：\n{context_json}\n"
        )
        from rich.console import Console as _Console
        _status_console = _Console()
        with _status_console.status("汇总中...", spinner="dots"):
            ans = summarizer_llm.invoke(prompt)
        return {"final_answer": ans.model_dump()}
    return _node

def build_app(llm: "ModelScopeChat"):
    """
    构建线性工作流：planner → executor → summarizer → END
    教学说明：
    - 我们使用 LangGraph 的 StateGraph 来定义“节点 + 边”的有状态流程图
    - 每个节点（planner/executor/summarizer）都是一个可调用的函数，接收并返回状态（ToolUseState 的部分字段）
    - set_entry_point 指定流程的起点；add_edge 用于指定节点之间的执行顺序
    - compile 将构建器“编译”为可执行的应用对象（app），随后可用 app.stream 进行流式执行
    """
    graph_builder = StateGraph(ToolUseState)  # 声明状态类型，便于 IDE 与读者理解节点间传递的字段结构
    # 添加 3 个核心节点：规划 → 执行 → 汇总
    graph_builder.add_node("planner", make_planner_node(llm))       # 让 LLM 生成结构化的工具调用计划（ToolPlan）
    graph_builder.add_node("plan_check", make_plan_check_node())    # 规划自检与规范化，降低执行阶段风险
    graph_builder.add_node("executor", make_executor_node())        # 依次执行计划，累积上下文字典（context）
    graph_builder.add_node("summarizer", make_summarizer_node(llm)) # 根据执行结果与上下文生成最终回答（FinalAnswer）
    # 设置流程入口与边（线性编排）
    graph_builder.set_entry_point("planner")        # 入口：先做“规划”
    graph_builder.add_edge("planner", "plan_check") # 边：规划 → 规划自检
    graph_builder.add_edge("plan_check", "executor")# 边：自检 → 执行
    graph_builder.add_edge("executor", "summarizer")# 边：执行 → 汇总
    graph_builder.add_edge("summarizer", END)      # 结束：汇总后结束（返回最终状态）
    # compile 将图转为可运行对象；你可以用 app.invoke/app.stream 执行
    return graph_builder.compile()

def run_workflow(app, user_request: str) -> ToolUseState:
    """
    执行工作流并返回最终状态；打印状态字段变化便于学习理解
    教学说明：
    - initial_input 是工作流的初始状态，只包含用户请求（user_request）
    - app.stream 会按我们在图中定义的“节点顺序”执行，并在每一步返回当前的“状态增量”
    - stream_mode="values" 表示仅返回状态字典（不包含节点名等额外信息），便于直观查看字段变化
    - final_state 持有最后一步的完整状态，通常会包含 tool_plan、tool_results、final_answer 等字段
    """
    initial_input = {"user_request": user_request}  # 初始状态：仅含用户请求文本
    final_state: Optional[ToolUseState] = None
    # 逐步执行：planner → executor → summarizer → END
    for state_update in app.stream(initial_input, stream_mode="values"):
        final_state = state_update  # 覆盖为当前步骤的最新状态；循环结束后即为最终状态
        if DEBUG:
            # 打印该步骤产生/更新的状态字段名，帮助新手观察“数据是如何逐步丰富起来的”
            console.print(f"[bold]状态更新：[/bold]{list(state_update.keys())}")
    # 若因异常中断，则兜底返回 initial_input；正常则返回完整 final_state
    return final_state or initial_input

def print_outputs(state: ToolUseState) -> None:
    """打印工具计划、执行结果与最终回答，便于直观教学展示"""
    console.print("--- ### 工具计划 ---")
    plan = state.get("tool_plan", {})
    if plan:
        console.print(json.dumps(plan, ensure_ascii=False, indent=2))
    console.print("--- ### 执行结果 ---")
    results = state.get("tool_results", {})
    if results:
        console.print(json.dumps(results, ensure_ascii=False, indent=2))
    console.print("--- ### 最终回答 ---")
    final_ans = state.get("final_answer", {}).get("answer", "")
    if final_ans:
        console.print(final_ans)

def parse_args() -> argparse.Namespace:
    """命令行参数解析：支持请求文本与教学日志开关"""
    parser = argparse.ArgumentParser(description="工具使用（Tool Use）架构：规划→执行→汇总的可运行脚本")
    parser.add_argument(
        "--request",
        type=str,
        default="请对这段话做简单文本管线：'LangGraph makes it easier to build stateful AI workflows.' 标准化、分词、提取5个关键词，最后结合当前时间渲染为 Markdown 报告。",
        help="用户请求（建议写明文本与简化的管线步骤）",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="开启详细日志",
    )
    parser.add_argument(
        "--stream",
        action="store_true",
        help="实时打印模型令牌（结构化 JSON 也将边接收边展示）",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=10,
        help="限制规划可执行的最大步骤数（过长的计划将被截断）",
    )
    parser.add_argument(
        "--on-error",
        type=str,
        choices=["stop", "skip", "fallback"],
        default="skip",
        help="执行阶段的错误策略：stop=停止，skip=跳过继续，fallback=继续并由汇总说明",
    )
    return parser.parse_args()

def main():
    """脚本入口：初始化环境变量与 LLM，构建工作流并运行"""
    load_dotenv()
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_PROJECT"] = "Agentic Architecture - Tool Use (ModelScope)"
    args = parse_args()
    global DEBUG
    DEBUG = bool(args.debug)
    global STREAM_TOKENS
    STREAM_TOKENS = bool(args.stream)
    global MAX_STEPS, ON_ERROR
    try:
        MAX_STEPS = int(args.max_steps)
    except Exception:
        MAX_STEPS = 10
    ON_ERROR = str(args.on_error or "skip")
    if not os.environ.get("MODELSCOPE_API_KEY"):
        console.print("[bold red]MODELSCOPE_API_KEY 未设置，请在项目根目录配置 .env[/bold red]")
    if not os.environ.get("LANGCHAIN_API_KEY"):
        console.print("[bold yellow]提示：未设置 LANGCHAIN_API_KEY，LangSmith 追踪将不可用[/bold yellow]")
    llm = init_llm()
    if DEBUG:
        console.print("[bold cyan]推理服务配置[/bold cyan]:")
        console.print(f"base_url={llm.base_url}")
        console.print(f"model_id={llm.model}")
    app = build_app(llm)
    final_state = run_workflow(app, args.request)
    print_outputs(final_state)

if __name__ == "__main__":
    main()

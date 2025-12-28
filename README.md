# AgenticArchitectures

面向学习与实践的“智能体架构”示例集合。当前包含四套核心示例：

- 反思（Reflection）架构：`01_reflection.py`
- 工具使用（Tool Use）架构：`02_tool_use.py`
- ReAct（Reasoning + Acting）架构：`03_react.py`
- 规划（Planning）架构：`04_planning.py`

目标：帮助你理解如何用 LangGraph 1.0 构建有状态工作流、用 Pydantic v2 约束 LLM 结构化输出，并通过命令行可运行的脚本体验端到端流程。

## 环境准备
- 建议 Python 3.10+
- 创建虚拟环境并安装依赖：
  - `python -m venv .venv`
  - `.\.venv\Scripts\activate`
  - `pip install -r requirements.txt`
- 在项目根目录创建 `.env`，配置推理与追踪相关环境变量：
  - `MODELSCOPE_API_KEY`（必需，ModelScope 推理令牌）
  - `MODELSCOPE_BASE_URL`（可选，默认 `https://api-inference.modelscope.cn/v1`）
  - `MODELSCOPE_MODEL_ID`（可选，默认 `deepseek-ai/DeepSeek-V3.2`）
  - `LANGCHAIN_API_KEY`（可选，用于 LangSmith 追踪）

## 反思（Reflection）架构
文件：`01_reflection.py`

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐     ┌───────┐
│  生成初稿   │────▶│   评审      │────▶│   改写      │────▶│  结束 │
│  generator  │     │  critic     │     │  refiner    │     │  END  │
└─────────────┘     └─────────────┘     └─────────────┘     └───────┘
```

- 核心流程：生成初稿 → 评审 → 改写（结构化输出）
- 运行：
  - `python 01_reflection.py`
  - 自定义请求：`python 01_reflection.py --request "Write a Python function to sort a list."`
  - 保存改写代码：`python 01_reflection.py --save-refined refined.py`
  - 教学日志：`python 01_reflection.py --debug`
- 要点：
  - 使用 `ModelScope` 的 OpenAI 兼容接口，并通过适配器 `with_structured_output` 要求模型以 JSON 结构返回
  - 用 `Pydantic v2` 定义数据模型（如 `DraftCode`、`Critique`、`RefinedCode`）并进行校验
  - 基于 `LangGraph.StateGraph` 编排线性工作流：`generator → critic → refiner → END`

## 工具使用（Tool Use）架构
文件：`02_tool_use.py`

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│    规划     │────▶│    执行     │────▶│    汇总     │
│  planner    │     │  executor   │     │ synthesizer │
└─────────────┘     └─────────────┘     └─────────────┘
```

- 核心流程：规划 → 执行 → 汇总（结构化输出）
- 运行：
  - `python 02_tool_use.py`
  - 教学日志：`python 02_tool_use.py --debug`
  - 实时令牌打印：`python 02_tool_use.py --stream`
  - 运行策略：
    - 限制步数：`--max-steps 8`
    - 错误策略：`--on-error stop|skip|fallback`
  - 自定义请求（建议写明要处理文本与希望的管线步骤）：
    - `python 02_tool_use.py --request "请对文本做标准化、分词、提取5个关键词、记录时间并生成 Markdown 报告"`
- 设计要点：
  - 可串联工具的“文本处理管线”（精简版），例如：
    - `normalize_text` → `tokenize_words` → `keyword_extract` → `current_time` → `render_report`
  - 规划自检节点：在执行前进行规范化与校验，降低执行阶段风险（步数截断、未知工具跳过、变量引用提示）
  - 规划层输出 `ToolPlan`，每步可通过 `assign_to` 将结果保存到上下文变量中，后续步骤用 `"$变量名"` 或 `"$变量名.字段"` 引用
  - 执行器按序解析变量并执行工具，最终将 `context` 一并返回，便于教学展示
  - 汇总层使用 `FinalAnswer` 结构，将结果与来源汇总输出（并展示上下文）

## 常见问题
- 未设置 `MODELSCOPE_API_KEY`
  - 运行时会有红色提示；请在 `.env` 中配置有效令牌
- 规划 JSON 字段不匹配（如返回 `name/args/assign` 等别名）
  - 适配器已内置常见别名映射与归一化处理；若仍异常，可开启 `--debug` 查看原始返回并调整请求描述
- 实时日志
  - 使用 `--stream` 观察结构化 JSON 的令牌流；规划与汇总阶段会显示“规划中/汇总中”状态与实时输出

## ReAct（Reasoning + Acting）架构
文件：`03_react.py`

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│    思考     │────▶│    行动     │────▶│    观察     │
│  reasoning  │     │   action    │     │ observation │
└─────┬───────┘     └─────────────┘     └─────┬───────┘
      │                                        │
      └────────────────────────────────────────┘
```

- 核心流程：思考 → 行动 → 观察 → 思考...（循环）
- 运行：
  - `python 03_react.py`
- 要点：
  - 实现了经典的 ReAct 智能体架构，结合了推理和行动能力
  - 使用 LangGraph 构建循环工作流，直到任务完成
  - 支持工具调用和结果观察，实现动态决策

## 规划（Planning）架构
文件：`04_planning.py`

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│    规划     │────▶│    执行     │────▶│    合成     │
│  planner    │     │  executor   │     │ synthesizer │
└─────────────┘     └─────────────┘     └─────────────┘
```

- 核心流程：规划 → 执行 → 合成
- 运行：
  - `python 04_planning.py`
- 要点：
  - 在执行前先将复杂任务分解为子目标
  - 使用规划器生成详细步骤，执行器按步骤执行，合成器生成最终答案
  - 与 ReAct 架构进行对比，展示不同智能体架构的特点

## 目录结构（简要）
- `01_reflection.py` 反思架构示例
- `02_tool_use.py` 工具使用架构示例（含多步文本管线工具）
- `03_react.py` ReAct（推理+行动）架构示例
- `04_planning.py` 规划架构示例
- `requirements.txt` 运行依赖
- `LICENSE` 开源协议

## 许可
本项目遵循仓库内的 `LICENSE`。在使用与改造时请遵守相关条款。

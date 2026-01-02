# Agentic Architectures 智能体架构示例集

一个展示多种智能体架构的可运行示例集合，使用 LangGraph 和 ModelScope API 构建。

## 📋 项目概述

本项目提供了一系列智能体架构的完整实现，帮助您理解和学习不同的智能体设计模式。每个示例都包含完整的代码、注释和使用说明，方便您快速上手和扩展。

## 🎯 学习目标

通过本项目，您将能够：
- 理解不同智能体架构的工作原理
- 掌握 LangGraph 构建有状态工作流的方法
- 学会使用 Pydantic v2 进行结构化输出
- 了解如何与 ModelScope API 集成
- 掌握智能体错误处理和模型自动切换技术

## 📁 项目结构

```
.
├── 01_reflection.py               # 反思（Reflection）架构示例
├── 02_tool_use.py                 # 工具使用（Tool Use）架构示例
├── 03_react.py                    # ReAct（Reasoning + Action）架构示例
├── 04_planning.py                 # 规划（Planning）架构示例
├── 05_multi_agent.py              # 多智能体（Multi-Agent）架构示例
├── 06_planner_executor_verifier.py  # 规划→执行→验证（Planner→Executor→Verifier）架构示例
├── agentic_architecture_visualizer.py  # 架构可视化工具
├── .env.example                   # 环境变量示例文件
└── README.md                      # 项目说明文档
```

## 🚀 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置环境变量

在项目根目录创建 `.env` 文件，并添加以下配置：

```
# ModelScope API 配置
MODELSCOPE_BASE_URL=https://api-inference.modelscope.cn/v1
MODELSCOPE_API_KEY=your_api_key_here
MODELSCOPE_MODEL_ID=deepseek-ai/DeepSeek-V3.2
MODELSCOPE_MODEL_ID_R1=deepseek-ai/DeepSeek-R1-0528  # 备用模型（可选）

# LangSmith 配置（可选，用于追踪）
LANGCHAIN_API_KEY=your_langchain_api_key_here
LANGCHAIN_TRACING_V2=true
LANGCHAIN_PROJECT=Agentic Architecture
```

### 3. 运行示例

每个架构都可以独立运行：

```bash
# 运行反思架构示例
python 01_reflection.py

# 运行工具使用架构示例
python 02_tool_use.py

# 运行 ReAct 架构示例
python 03_react.py

# 运行规划架构示例
python 04_planning.py

# 运行多智能体架构示例
python 05_multi_agent.py

# 运行规划→执行→验证架构示例
python 06_planner_executor_verifier.py

# 运行架构可视化工具
streamlit run agentic_architecture_visualizer.py
```

## 📚 架构详解

### 1. 反思（Reflection）架构

**文件**: `01_reflection.py`

**核心思想**: 生成 → 评审 → 改写

- 先生成初步结果
- 然后对结果进行评审，找出问题和改进点
- 最后根据评审意见进行改写，得到更可靠的结果

**使用示例**:
```bash
python 01_reflection.py --request "Write a Python function to find the nth Fibonacci number."
```

### 2. 工具使用（Tool Use）架构

**文件**: `02_tool_use.py`

**核心思想**: 规划 → 执行 → 汇总

- 根据用户请求规划工具使用步骤
- 执行工具调用，获取外部信息
- 汇总结果，生成最终回答

**使用示例**:
```bash
python 02_tool_use.py --request "请计算表达式 2+3*4，并列出当前目录文件。"
```

### 3. ReAct（Reasoning + Action）架构

**文件**: `03_react.py`

**核心思想**: 思考 → 行动 → 观察 → 再思考

- 通过循环：思考下一步行动 → 执行行动 → 观察结果 → 调整思考
- 适合需要多轮交互和推理的任务

**使用示例**:
```bash
python 03_react.py
```

### 4. 规划（Planning）架构

**文件**: `04_planning.py`

**核心思想**: 任务分解 → 子任务执行 → 结果合成

- 首先将复杂任务分解为多个子任务
- 然后依次执行每个子任务
- 最后将子任务结果合成为最终答案

**使用示例**:
```bash
python 04_planning.py
```

### 5. 多智能体（Multi-Agent）架构

**文件**: `05_multi_agent.py`

**核心思想**: 智能体协作 → 任务分配 → 结果整合

- 多个智能体协同工作
- 根据各自专长分配任务
- 整合所有智能体的结果

**使用示例**:
```bash
python 05_multi_agent.py
```

### 6. 规划→执行→验证（Planner→Executor→Verifier）架构

**文件**: `06_planner_executor_verifier.py`

**核心思想**: 规划 → 执行 → 验证 → 迭代

- 规划器：将任务分解为可执行的步骤
- 执行器：执行规划的步骤并获取结果
- 验证器：检查执行结果是否符合预期
- 迭代机制：如果验证失败，返回重新规划

**使用示例**:
```bash
python 06_planner_executor_verifier.py
```

**关键特性**:
- 错误检测与自修正能力
- 基于条件的工作流路由
- 结构化的任务规划和执行
- 支持复杂任务的多轮迭代

## 🎨 架构可视化

**文件**: `agentic_architecture_visualizer.py`

提供了一个基于Streamlit的可视化界面，展示所有架构的工作流程和组件关系。

**界面预览**:

![架构可视化界面](AgenticArchitecture.jpg)

**使用示例**:
```bash
streamlit run agentic_architecture_visualizer.py
```

## 🔧 核心功能

### 模型自动切换

当主模型（`MODELSCOPE_MODEL_ID`）因API请求失败（如每天使用次数限制、网络错误等）而无法使用时，系统会自动切换到备用模型（`MODELSCOPE_MODEL_ID_R1`），确保工作流的连续性。切换逻辑实现了防无限循环机制，确保系统稳定性。

### 结构化输出

使用 Pydantic v2 约束 LLM 输出为结构化数据，提高输出的可靠性和可用性。

### 有状态工作流

使用 LangGraph 构建有状态的工作流，支持复杂的多步骤智能体逻辑。

### 友好的日志输出

提供详细的日志和可视化输出，方便理解智能体的工作过程和调试。

## 📝 命令行参数

每个示例都支持一些命令行参数，例如：

- `--request`：指定用户请求
- `--debug`：开启调试模式
- `--save-refined`：保存改写后的代码（仅反思架构）

使用 `--help` 查看详细参数：

```bash
python 01_reflection.py --help
```

## 🛠️ 技术栈

- Python 3.8+
- LangGraph：构建有状态工作流
- Pydantic v2：结构化输出
- ModelScope API：大语言模型接口
- Rich：终端美化输出

## 📄 许可证

MIT License

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

## 📧 联系方式

如有问题或建议，请通过 GitHub Issues 与我们联系。

---

**开始您的智能体架构之旅吧！** 🚀
# AgenticArchitectures

面向学习与实践的“智能体架构”示例集合。当前包含五套核心示例：

- 反思（Reflection）架构：`01_reflection.py`
- 工具使用（Tool Use）架构：`02_tool_use.py`
- ReAct（Reasoning + Acting）架构：`03_react.py`
- 规划（Planning）架构：`04_planning.py`
- 多智能体（Multi-Agent）系统：`05_multi_agent.py`

目标：帮助你理解如何用 LangGraph 1.0 构建有状态工作流、用 Pydantic v2 约束 LLM 结构化输出，并通过命令行可运行的脚本或可视化界面体验端到端流程。

## 可视化功能

项目提供了基于 Streamlit 的交互式可视化界面，当前支持多智能体系统的可视化分析：

- 文件：`05_multi_agent_visualization.py`
- 功能：直观的公司选择、实时分析进度显示、结构化结果展示、单/多智能体系统对比
- 运行：`streamlit run 05_multi_agent_visualization.py`

> 未来计划扩展到其他智能体架构的可视化支持。

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

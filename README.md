# Agentic Architectures 智能体架构示例集

一个展示 **17 种**智能体架构的可运行示例集合，使用 LangGraph 和 ModelScope API 构建。

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
├── 07_blackboard.py               # 黑板系统（Blackboard System）架构示例
├── 08_episodic_with_semantic_cn.py # 情景记忆 + 语义记忆栈（Episodic + Semantic Memory Stack）架构示例
├── 09_tree_of_thoughts_cn.py      # 思维树（Tree-of-Thoughts）架构示例
├── 10_mental_loop_cn.py           # 思维模型循环（Mental-Model-in-the-Loop）架构示例
├── 11_meta_controller_cn.py       # 元控制器（Meta-Controller）架构示例
├── 12_graph_cn.py                 # 图/世界模型记忆（Graph）架构示例
├── 13_ensemble_cn.py             # 并行探索 + 集成决策（Ensemble）架构示例
├── 14_dry_run_cn.py               # 可观测与试跑外壳（Dry-Run Harness）架构示例
├── 15_RLHF_cn.py                  # 自改进循环（Self-Refine / RLHF 类比）架构示例
├── 16_cellular_automata_cn.py     # 细胞自动机/网格智能体（Cellular Automata）架构示例
├── 17_reflexive_metacognitive_cn.py # 反思式元认知（Reflexive Metacognitive）架构示例
├── agentic_architecture_visualizer.py  # 架构可视化工具
├── agentic_recommend.py            # 架构推荐器：根据需求推荐并运行
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
# 备用模型（主模型 429/失败时自动切换，同 base_url 下换为 R1）
MODELSCOPE_MODEL_ID_R1=deepseek-ai/DeepSeek-R1-0528

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

# 运行黑板系统架构示例
python 07_blackboard.py

# 运行情景记忆 + 语义记忆栈架构示例
python 08_episodic_with_semantic_cn.py

# 运行思维树架构示例
python 09_tree_of_thoughts_cn.py

# 运行思维模型循环架构示例
python 10_mental_loop_cn.py

# 运行元控制器架构示例
python 11_meta_controller_cn.py

# 运行图/世界模型记忆架构示例（知识图谱构建与多跳问答）
python 12_graph_cn.py
# 自定义问题：python 12_graph_cn.py --question "谁在 AlphaCorp 工作？"
# 仅构建图：python 12_graph_cn.py --ingest-only

# 运行并行探索+集成决策架构示例（投资委员会）
python 13_ensemble_cn.py
# 自定义问题：python 13_ensemble_cn.py --request "英伟达 2026 中长期是否值得投资？"

# 运行可观测与试跑外壳架构示例（拟发帖→试跑→审核→执行/取消）
python 14_dry_run_cn.py
# 自定义请求：python 14_dry_run_cn.py --request "为我们的新 AI 产品写一条发布公告"

# 运行自改进循环架构示例（生成→评审→修订）
python 15_RLHF_cn.py
# 自定义请求：python 15_RLHF_cn.py --request "为我们新的 AI 数据分析平台写一封营销邮件"

# 运行细胞自动机/网格智能体架构示例（仓库拣货）
python 16_cellular_automata_cn.py
# 指定拣货清单：python 16_cellular_automata_cn.py --order A,B

# 运行反思式元认知架构示例（医疗分诊）
python 17_reflexive_metacognitive_cn.py
# 自定义问题：python 17_reflexive_metacognitive_cn.py --request "布洛芬和赖诺普利能一起吃吗？"

# 运行架构可视化工具
streamlit run agentic_architecture_visualizer.py

# 架构推荐器：根据需求推荐并运行
python agentic_recommend.py recommend "我需要医疗分诊，能识别急症转人工"
python agentic_recommend.py run 17 --request "布洛芬和赖诺普利能一起吃吗？"
python agentic_recommend.py list
```

## 🔍 架构推荐器（agentic_recommend.py）

根据需求描述，**由 LLM 智能推荐**最合适的智能体架构。各架构 py 工具有限，不能直接跑用户的具体需求，需**先跑该架构自带示例**，看懂架构后再二创落地。

| 命令 | 说明 |
|------|------|
| `recommend "需求描述"` | LLM 理解需求后推荐架构 + 贴心理由 + 运行该架构自带示例的命令 |
| `run <编号> [参数]` | 运行指定架构，参数透传给对应 py 脚本 |
| `list` | 列出全部 17 种架构 |

**核心能力**：
- **智能推荐**：LLM 理解产品需求，从 17 种架构中选出最合适的一种
- **正确命令**：系统根据各架构的实际 CLI 参数（如 03 用 `--question`、05 用 `--company`、16 用 `--order`）自动生成可运行命令，避免参数名错误
- **落地说明**：推荐结果明确提示「各架构工具有限，先跑自带示例，看懂架构后再二创接入自己的工具、数据、业务逻辑落地产品」

**示例**：
```bash
python agentic_recommend.py recommend "我需要医疗分诊，能识别急症转人工"
python agentic_recommend.py recommend "我想做个点咖啡智能体，根据习惯选门店、选品、下单送达"
python agentic_recommend.py recommend "给我们的 AI 产品写营销邮件，要迭代打磨"
python agentic_recommend.py list
```

**真实运行效果**（点咖啡需求）示例：
```
$ python agentic_recommend.py recommend "我想做个点咖啡智能体，可以根据我的习惯进行选门店、选品、然后下单送达"

根据「我想做个点咖啡智能体...」的智能推荐：

  [推荐] #03 ReAct

  [理由] 推荐ReAct架构，因为它支持多步推理和工具调用循环……

  [运行] python 03_react.py

  [说明] 各架构 py 里的工具有限，不能直接跑你的点咖啡需求。先跑上面的命令，运行该架构自带的示例，看懂思考→行动→观察循环怎么工作，然后才知道怎么二创、接入门店 API、下单接口等自己的工具与数据。
```

**重要**：各架构自带的工具有限（如 03 的计算器、列目录），不能直接跑用户的具体需求。**先跑该架构自带示例**（如 `python 03_react.py`），看懂架构后再二创落地。

**运行命令**：
```bash
python agentic_recommend.py run 17 --request "布洛芬和赖诺普利能一起吃吗？"
python agentic_recommend.py run 3
python agentic_recommend.py run 16 --order A,B --verbose
```

**前置条件**：推荐功能需配置 `.env` 中的 `MODELSCOPE_API_KEY`。

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

### 7. 黑板系统（Blackboard System）架构

**文件**: `07_blackboard.py`

**核心思想**: 共享内存 → 专家智能体 → 动态控制 → 机会主义激活

- 共享内存（黑板）：中央数据存储，所有智能体可以读取和写入
- 专家智能体：具有特定专业知识的独立智能体
- 动态控制器：观察黑板状态并决定下一个执行的智能体
- 机会主义激活：智能体根据当前问题状态被动态激活

**使用示例**:
```bash
# 直接运行默认示例
python 07_blackboard.py

# 自定义查询
python 07_blackboard.py --query "分析最近的阿里巴巴新闻并生成投资建议"

# 跳过线性系统运行
python 07_blackboard.py --no-sequential

# 启用调试模式
python 07_blackboard.py --debug
```

**关键特性**:
- 高度灵活的多智能体协调模式
- 动态控制器决定执行顺序
- 支持与线性多智能体系统对比
- 可选的真实网络搜索功能
- 自动回退到模拟搜索工具

### 8. 情景记忆 + 语义记忆栈（Episodic + Semantic Memory Stack）架构

**文件**: `08_episodic_with_semantic_cn.py`

**核心思想**: 情景记忆 → 语义记忆 → 记忆检索 → 增强生成

- 情景记忆：存储特定事件或过去交互的记忆，使用向量数据库管理
- 语义记忆：存储从事件中提取的结构化事实、概念和关系，使用图数据库管理
- 记忆检索：从两个记忆系统中查询相关信息
- 增强生成：结合检索到的记忆生成个性化响应

**使用示例**:
```bash
# 直接运行默认示例
python 08_episodic_with_semantic_cn.py
```

**关键特性**:
- 持久记忆系统，支持长期个性化交互
- 结合向量数据库和图数据库的优势
- 自动从对话中提取结构化知识
- 支持相似性搜索和关系查询
- 中文界面和详细注释

### 9. 思维树（Tree-of-Thoughts）架构

**文件**: `09_tree_of_thoughts_cn.py`

**核心思想**: 并行探索 → 路径评估 → 分支修剪 → 解决方案合成

- 并行探索：同时考虑多种可能的解决方案路径
- 路径评估：定期评估各路径的可行性和进展
- 分支修剪：剪枝无效或进展缓慢的路径
- 解决方案合成：综合最优路径生成最终解决方案

**使用示例**:
```bash
# 直接运行默认示例
python 09_tree_of_thoughts_cn.py

# 自定义问题
python 09_tree_of_thoughts_cn.py --problem "从数字1开始，使用+1、×3、-2操作，在10步内到达数字31"
```

**关键特性**:
- 支持多路径并行探索
- 动态路径评估和剪枝
- 可视化解决方案路径
- 中文界面和详细注释

### 10. 思维模型循环（Mental-Model-in-the-Loop）架构

**文件**: `10_mental_loop_cn.py`

**核心思想**: 观察 → 提议 → 模拟 → 评估 → 执行

- 观察：观察真实环境的当前状态
- 提议：基于当前状态生成高级策略
- 模拟：在沙盒环境中运行策略模拟
- 评估：分析模拟结果，评估风险和回报
- 执行：在真实环境中执行经过优化的策略

**使用示例**:
```bash
# 直接运行默认示例
python 10_mental_loop_cn.py
```

### 11. 元控制器（Meta-Controller）架构

**文件**: `11_meta_controller_cn.py`

**核心思想**: 接收请求 → 元控制器分析并路由 → 专家执行 → 返回结果

- 元控制器：分析用户请求的意图与类型，决定路由到哪个专家
- 专家智能体：通用（日常对话）、研究（搜索与复杂问题）、编码（Python 代码）
- 单一入口、按请求类型路由到最合适的专家

**使用示例**:
```bash
# 直接运行（脚本内带多组测试请求）
python 11_meta_controller_cn.py
```

**关键特性**:
- 智能路由，无需固定流水线
- 易扩展新专家（增加节点并更新路由逻辑）
- 研究专家可搭配 Tavily 搜索工具

### 12. 图/世界模型记忆（Graph）架构

**文件**: `12_graph_cn.py`

**核心思想**: 文本 → 知识图谱抽取 → 写入图 → 自然语言问题 → Cypher 查询 → 图执行 → 合成答案

- 图构建：从非结构化文本抽取实体与关系（LLM + 结构化输出），写入内存图或 Neo4j
- Text-to-Cypher：根据图 schema 与用户问题生成 Cypher，执行后合成自然语言答案
- 支持多跳推理（如「收购 BetaSolutions 的公司所生产产品的竞品来自哪家公司？」）

**使用示例**:
```bash
# 直接运行（默认摄入 3 段文档并回答 3 个示例问题）
python 12_graph_cn.py
# 自定义问题
python 12_graph_cn.py --question "谁在 AlphaCorp 工作？"
# 仅构建图不问答
python 12_graph_cn.py --ingest-only
```

**关键特性**:
- 无 Neo4j 时使用内存图演示；可选 `NEO4J_URI` 等连接真实图库
- 主模型 429/失败时自动切换备用（可配 `MODELSCOPE_MODEL_ID_R1`，同 API 下换为 DeepSeek-R1）

### 13. 并行探索+集成决策（Ensemble）架构

**文件**: `13_ensemble_cn.py`

**核心思想**: 多路分析师并行分析 → CIO 综合决策

- 三路分析师（看多成长/谨慎价值/量化）对同一问题独立分析，可选 Tavily 检索
- CIO 综合多份报告，输出结构化投资建议（Strong Buy/Buy/Hold/Sell/Strong Sell、信心分数、机会与风险列表）
- 适用：复杂推理、事实核查、高 stakes 决策支持

**使用示例**:
```bash
python 13_ensemble_cn.py
python 13_ensemble_cn.py --request "英伟达 2026 中长期是否值得投资？"
```

### 14. 可观测与试跑外壳（Dry-Run Harness）架构

**文件**: `14_dry_run_cn.py`

**核心思想**: 拟稿 → 试跑预览 → 人工审核（approve/reject）→ 执行或取消

- 工具支持 `dry_run=True` 时只打日志、不产生真实副作用
- 人机审核：将试跑结果展示给操作员，仅当输入 approve 时才真实执行
- 适用：发帖、发邮件、改数据库等不可逆操作的前置校验

**使用示例**:
```bash
python 14_dry_run_cn.py
python 14_dry_run_cn.py --request "为我们的新 AI 产品写一条发布公告"
```

### 15. 自改进循环（Self-Refine / RLHF 类比）架构

**文件**: `15_RLHF_cn.py`

**核心思想**: 生成 → 评审 → 未通过则修订再评审，直到通过或达最大轮数（默认 3 轮）

- 营销邮件初稿由评审打分（8 分及以上通过），未通过则根据反馈修订后再次评审
- 可选：将通过的样本写入「金标记忆」供后续任务参考（RLHF 类比）
- 适用：营销邮件、法律/技术文档等需高质量单篇输出的任务

**使用示例**:
```bash
python 15_RLHF_cn.py
python 15_RLHF_cn.py --request "为我们新的 AI 数据分析平台写一封营销邮件"
```

### 16. 细胞自动机/网格智能体（Cellular Automata）架构

**文件**: `16_cellular_automata_cn.py`

**核心思想**: 大量简单单元 + 局部规则，涌现路径规划等全局行为

- 从打包站（P）将 pathfinding_value=0 向外扩散形成梯度，从货架格沿梯度下降走到打包站即拣货路径
- 主要用 numpy + rich，无需 MODELSCOPE_API_KEY 即可跑网格演示；可选结尾 LLM 总结
- 适用：仓库拣货、空间推理、可并行局部更新的仿真

**使用示例**:
```bash
python 16_cellular_automata_cn.py
python 16_cellular_automata_cn.py --order A,B
# 智能体群组演示：python 16_cellular_automata_cn.py --verbose
```

**运行效果示例**（Rich 美化输出）:
```
--- 初始网格 ---
┌─┬─┬─┬─┬─┬─┬─┬─┬─┬─┬─┬─┬─┬─┬─┬─┬─┬─┬─┬─┬─┬─┬─┬─┬─┬─┐
│·│·│█│·│·│·│█│·│·│·│█│·│·│·│█│·│·│·│█│·│·│·│█│·│·│█│
│·│·│█│·│·│·│·│·│D│·│·│·│·│·│5│·│·│·│·│·│6│·│·│·│█│·│
│·│·│█│·│·│·│A│·│4│·│·│3│·│·│2│·│·│C│·│·│6│·│·│7│·│·│
│·│·│█│·│·│·│5│·│·│4│·│·│3│·│·│4│·│·│5│·│·│B│·│8│·│·│
│·│·│█│·│·│6│·│·│5│·│·│4│·│·│P│·│·│6│·│·│7│·│·│8│·│·│
│·│·│█│·│·│·│█│·│·│·│█│·│·│·│█│·│·│·│█│·│·│·│█│·│·│█│
└─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┘

--- 拣货清单：['A', 'B'] ---
拣货：A → 路径 (2,6)->(3,6)->...->(4,14)
拣货：B → 路径 (3,21)->...->(4,14)
[完成] 拣货完成：订单 ['A','B'] 已通过网格路径波完成拣货。

--- LLM 总结 ---
仓库网格拣货系统已根据订单 ['A','B'] 完成拣货，商品 A 和 B 分别沿指定路径移动，最终均汇合于点 (4,14)。
```

### 17. 反思式元认知（Reflexive Metacognitive）架构

**文件**: `17_reflexive_metacognitive_cn.py`

**核心思想**: 先对请求做元认知分析，再选策略：直接回答 / 用工具 / 升级人工

- 自模型：知识域、可用工具、置信度阈值；元认知分析输出策略与理由
- 演示：简单问题直接答、药物相互作用用 drug_interaction_checker、高危或超域则升级
- 适用：医疗/法律/金融等高风险领域，必须能说「请咨询专业人士」

**使用示例**:
```bash
python 17_reflexive_metacognitive_cn.py
python 17_reflexive_metacognitive_cn.py --request "布洛芬和赖诺普利能一起吃吗？"
```

## 🎯 架构对比：ReAct vs 思维模型循环

### 核心设计理念
- **ReAct**：「思考-行动-观察」的闭环循环，强调实时推理与工具使用的结合
- **思维模型循环**：「先思考后行动」的安全机制，强调风险评估与策略优化

### 工作流程
- **ReAct**：思考 → 行动 → 观察 → 思考 → 行动 → ... → 结束
  - 动态循环，条件边决定是否继续
  - 迭代次数有上限，防止无限循环

- **思维模型循环**：提出策略 → 运行模拟 → 完善决策 → 真实执行 → 结束
  - 线性流程，包含明确的四个阶段
  - 无迭代循环，一次完整流程解决一个问题

### 关键组件
- **ReAct**：
  - Reasoner（思考节点）：分析问题和历史观察，决定下一步行动
  - Actor（行动节点）：执行选定工具，记录观察结果
  - 条件边：根据状态决定继续循环或结束

- **思维模型循环**：
  - 分析师：基于市场状态提出高级交易策略
  - 模拟器：在沙盒环境中运行策略，生成多个模拟场景
  - 风险经理：分析模拟结果，完善决策降低风险
  - 执行器：在真实市场中执行最终决策

### 决策机制
- **ReAct**：基于工具执行的实时观察结果逐步调整策略，适合需要逐步探索的问题
- **思维模型循环**：基于多场景模拟的统计分析来完善策略，考虑多种可能的未来结果

### 适用场景
- **ReAct**：通用问题解决，如信息查询、计算、文本处理等需要使用外部工具的低风险任务
- **思维模型循环**：高风险环境，如金融交易、机器人操作等需要提前评估风险的场景

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

当主模型（`MODELSCOPE_MODEL_ID`，默认 DeepSeek-V3.2）因 API 请求失败（如 429 限流、每日配额、网络错误等）而无法使用时，系统会自动切换到备用模型（`MODELSCOPE_MODEL_ID_R1`，默认 DeepSeek-R1-0528），确保工作流的连续性。仅支持同 base_url 下换模型，切换逻辑实现了防无限循环机制，确保系统稳定性。

### 结构化输出

使用 Pydantic v2 约束 LLM 输出为结构化数据，提高输出的可靠性和可用性。

### 有状态工作流

使用 LangGraph 构建有状态的工作流，支持复杂的多步骤智能体逻辑。

### 友好的日志输出

提供详细的日志和可视化输出，方便理解智能体的工作过程和调试。

## 📝 命令行参数

每个示例都支持一些命令行参数，例如：

- `--request`：指定用户请求
- `--question`：指定图问答问题（仅 12_graph_cn）
- `--ingest-only`：仅构建图不问答（仅 12_graph_cn）
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

面向学习与实践的“智能体架构”示例集合。当前包含 **17 套**可运行示例（01–17），涵盖反思、工具使用、ReAct、规划、多智能体、黑板、记忆、思维树、元控制器、试跑外壳、自改进循环、细胞自动机、反思式元认知等架构。

目标：帮助你理解如何用 LangGraph 1.0 构建有状态工作流、用 Pydantic v2 约束 LLM 结构化输出，并通过命令行可运行的脚本或可视化界面体验端到端流程。

## 可视化功能

项目提供了基于 Streamlit 的交互式可视化界面，支持全部 17 种架构的可视化运行与演示：

- 文件：`agentic_architecture_visualizer.py`
- 功能：架构选择、参数配置、实时执行日志、结构化结果展示
- 运行：`streamlit run agentic_architecture_visualizer.py`

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
  - 高德MCP服务配置（可选，使用地图功能时需要）：
    - `AMAP_KEY`：在高德开放平台申请的API密钥（https://console.amap.com/）
    - `AMAP_MCP_URL`：高德MCP服务器地址（可选，默认：https://mcp.amap.com/mcp）

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
  - 支持高德MCP地图服务：可进行地理编码、逆地理编码、路线规划、兴趣点查询等地图相关操作

## 常见问题
- 未设置 `MODELSCOPE_API_KEY`
  - 运行时会有红色提示；请在 `.env` 中配置有效令牌
- 规划 JSON 字段不匹配（如返回 `name/args/assign` 等别名）
  - 适配器已内置常见别名映射与归一化处理；若仍异常，可开启 `--debug` 查看原始返回并调整请求描述
- 实时日志
  - 使用 `--stream` 观察结构化 JSON 的令牌流；规划与汇总阶段会显示“规划中/汇总中”状态与实时输出

## 高德MCP服务使用说明

### 功能介绍

工具使用架构中集成了高德MCP地图服务，可以实现以下功能：

- 地理编码（地址转坐标）
- 逆地理编码（坐标转地址）
- 路线规划
- 兴趣点查询
- 地图数据查询等

### 配置步骤

1. 在高德开放平台申请API密钥：https://console.amap.com/
2. 在 `.env` 文件中添加以下配置：

```ini
# 高德MCP地图服务配置
AMAP_KEY="您在高德开放平台申请的API密钥"
AMAP_MCP_URL="https://mcp.amap.com/mcp"  # 可选，默认值
```

### 使用示例

#### 示例1：地理编码（地址转坐标）

```bash
python 02_tool_use.py --request "请帮我查询北京市朝阳区的地理坐标"
```

#### 示例2：逆地理编码（坐标转地址）

```bash
python 02_tool_use.py --request "请帮我查询坐标116.407413,39.904211对应的地址"
```

#### 示例3：兴趣点查询

```bash
python 02_tool_use.py --request "请帮我查询北京市朝阳区附近的餐馆"
```

### 参数说明

高德MCP工具需要以下参数：

- `service`：高德MCP服务名称（如：`geocode`、`regeo`、`route`、`poi`等）
- `parameters`：服务对应的参数对象（根据不同服务类型提供不同参数）

系统会自动根据用户请求构建合适的参数并调用相应的高德MCP服务。

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

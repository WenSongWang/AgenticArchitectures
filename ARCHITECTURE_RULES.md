# Agentic Architectures 设计规范文档

本文档总结了项目中01~05 py文件的共同规则和最佳实践，旨在确保代码的一致性、可维护性和可扩展性。

## 1. 项目结构与命名规范

### 1.1 文件命名
- 按架构类型编号命名：`01_reflection.py`, `02_tool_use.py`, `03_react.py`, `04_planning.py`, `05_multi_agent.py`
- 通用模块命名：`utils.py`（存放共享组件和工具函数）
- 可视化工具命名：`agentic_architecture_visualizer.py`

### 1.2 类命名
- 使用驼峰命名法（CamelCase）：`ModelScopeChat`, `DraftCode`, `Critique`
- 类名应清晰反映其功能和职责

### 1.3 函数命名
- 使用蛇形命名法（snake_case）：`make_generator_node`, `build_app`, `run_workflow`
- 函数名应包含动词，清晰描述其行为

### 1.4 变量命名
- 使用蛇形命名法：`user_request`, `draft_code`, `critique_suggestions`
- 避免使用单字母变量（循环变量除外）
- 使用有意义的变量名，提高代码可读性

## 2. 代码结构与组织

### 2.1 文档字符串（Docstring）
每个文件都应包含详细的文档字符串，包括：
- 学习目标：明确说明读者通过本示例能学到什么
- 核心概念：速览本架构的关键概念
- 运行前准备：环境配置和依赖说明
- 如何运行：提供具体的运行命令和参数
- 阅读建议：指导读者如何高效阅读代码

### 2.2 代码模块划分
每个文件应按以下逻辑顺序组织代码：
1. **数据结构与模型定义**：使用Pydantic v2定义数据模型
2. **LLM初始化与配置**：设置语言模型客户端
3. **核心节点实现**：实现工作流的各个节点函数
4. **工作流构建**：使用LangGraph构建工作流
5. **运行与输出**：实现工作流的运行和结果展示
6. **命令行接口**：解析命令行参数
7. **主函数**：程序入口点

### 2.3 示例代码结构
```python
# 1. 数据结构与模型定义
class TaskModel(BaseModel):
    pass

class WorkflowState(TypedDict):
    pass

# 2. LLM初始化与配置
from utils import init_llm
llm = init_llm()

# 3. 核心节点实现
def make_node(llm):
    def _node(state):
        pass
    return _node

# 4. 工作流构建
def build_app(llm):
    graph = StateGraph(WorkflowState)
    graph.add_node("node", make_node(llm))
    return graph.compile()

# 5. 运行与输出
def run_workflow(app, input_data):
    pass

# 6. 命令行接口
def parse_args():
    pass

# 7. 主函数
def main():
    pass

if __name__ == "__main__":
    main()
```

## 3. 技术栈与依赖

### 3.1 核心技术栈
- **Python 3.8+**：基础编程语言
- **LangGraph**：构建有状态工作流
- **Pydantic v2**：结构化输出和数据验证
- **ModelScope API**：大语言模型接口（通过OpenAI兼容接口）
- **Rich**：终端美化输出
- **python-dotenv**：加载环境变量

### 3.2 环境变量配置
所有文件应使用以下环境变量：
- `MODELSCOPE_API_KEY`：ModelScope API密钥（必需）
- `MODELSCOPE_BASE_URL`：ModelScope API基础URL（默认：https://api-inference.modelscope.cn/v1）
- `MODELSCOPE_MODEL_ID`：默认模型ID（默认：deepseek-ai/DeepSeek-V3.2）
- `MODELSCOPE_MODEL_ID_R1`：备用模型ID（可选，用于自动切换）
- `LANGCHAIN_API_KEY`：LangSmith追踪密钥（可选）
- `LANGCHAIN_TRACING_V2`：启用LangSmith追踪（默认：true）
- `LANGCHAIN_PROJECT`：LangSmith项目名称（默认：Agentic Architecture - [架构名称]）

## 4. 工作流设计规范

### 4.1 LangGraph使用
- 使用`StateGraph`构建有状态工作流
- 明确定义工作流状态（`WorkflowState`）
- 每个节点应是一个独立的函数，接收状态并返回状态更新
- 使用`add_node`添加节点，使用`add_edge`定义节点间的连接
- 使用`set_entry_point`定义入口节点

### 4.2 结构化输出
- 使用Pydantic v2模型定义所有结构化数据
- 为每个模型添加详细的字段描述
- 使用`with_structured_output`包装LLM调用，确保返回结构化数据
- 实现字段映射和错误处理，提高解析成功率

### 4.3 节点设计
- 节点函数应接收当前状态并返回状态更新
- 使用闭包模式（如`make_generator_node`）封装节点逻辑，便于参数传递
- 节点间应通过状态共享数据，避免直接依赖

## 5. 错误处理与稳定性

### 5.1 模型自动切换
- 实现模型请求失败时的自动切换功能
- 使用`switched`标志防止无限切换
- 记录模型切换日志，便于调试

### 5.2 结构化输出错误处理
- 实现JSON解析错误的优雅处理
- 提供字段映射机制，处理常见的字段别名
- 记录解析错误和字段映射日志

### 5.3 输入验证
- 验证环境变量是否正确设置
- 验证用户输入的有效性
- 提供清晰的错误提示

## 6. 日志与调试

### 6.1 日志配置
- 使用`rich.logging.RichHandler`实现美化日志输出
- 为每个模块创建独立的日志记录器
- 设置日志级别（DEBUG/INFO）

### 6.2 调试模式
- 支持`--debug`命令行参数开启调试模式
- 调试模式下输出详细的中间结果和状态变化
- 调试模式下输出结构化提示和模型原始返回

### 6.3 控制台输出
- 使用`rich.console.Console`实现美化控制台输出
- 使用不同颜色区分不同类型的信息（成功、警告、错误）
- 对代码和JSON进行格式化输出，提高可读性

## 7. 命令行接口

### 7.1 参数设计
- 支持`--request`/`--question`参数：自定义用户请求
- 支持`--debug`参数：开启调试模式
- 支持`--stream`参数：启用令牌流输出
- 根据架构特点设计特定参数（如`--save-refined`）

### 7.2 参数解析
- 使用`argparse`模块解析命令行参数
- 提供清晰的参数描述和默认值
- 支持`--help`参数查看帮助信息

## 8. 性能与效率

### 8.1 模型调用优化
- 合理设置温度参数（默认：0.2）
- 根据需要启用/禁用流式输出
- 避免不必要的模型调用

### 8.2 内存管理
- 及时清理不再需要的中间结果
- 避免内存泄漏

## 9. 可维护性与扩展性

### 9.1 代码复用
- 将通用功能抽象到`utils.py`模块
- 避免代码重复
- 使用统一的接口和模式

### 9.2 注释
- 为复杂逻辑添加注释
- 解释代码的设计思路和决策
- 避免冗余注释

### 9.3 测试
- 提供清晰的使用示例
- 确保代码可运行和可复现
- 测试边界情况和错误处理

## 10. 安全考虑

### 10.1 API密钥管理
- 从环境变量加载API密钥，避免硬编码
- 不要将API密钥提交到版本控制系统
- 提示用户创建`.env`文件存储敏感信息

### 10.2 输入验证
- 验证用户输入的安全性
- 防止注入攻击

### 10.3 工具调用安全
- 仅允许调用安全的工具和函数
- 限制工具的权限和范围

---

遵循以上规范将有助于保持代码的一致性、可维护性和可扩展性，同时提高开发效率和代码质量。
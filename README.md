# LangChain 1.x Agent

用 **LangChain 1.x 的 `create_agent`** 实现的 Agent，作为 [react-agent-from-scratch](https://github.com/Linjian5/react-agent-from-scratch)（手写版）的**框架对照实现**——同样的 3 个工具、同样的 ReAct 范式，对比"手写 vs 框架"的代码量与控制粒度。

后端走 OpenAI 兼容协议，默认 [DeepSeek](https://platform.deepseek.com/)，改 `MODEL` + `OPENAI_BASE_URL` 即可切 OpenAI / Kimi。

## ⚡ LangChain 1.x 重构说明（重要）

LangChain 1.0（2025 年 GA）下线了旧 Agent API。本项目使用最新 API：

```python
# ❌ 旧版（0.x，已下线）
from langchain.agents import AgentExecutor, create_react_agent

# ✅ 新版（1.x）
from langchain.agents import create_agent
agent = create_agent(model=llm, tools=TOOLS, system_prompt="...")
# 返回的是 LangGraph CompiledStateGraph
```

旧版基于自研 AgentExecutor + ReAct 文本解析；新版**底层就是 LangGraph**——LangChain 团队把状态机编排统一到了 LangGraph 上。这意味着：

- 自动获得 LangGraph 的 `stream / astream / invoke`、checkpointing 接口、interrupt 中断点
- Tool 调用走 Function Calling（不再依赖 LLM 输出 ReAct 文本格式）
- 并行调用工具开箱即用（一次输出多个 `tool_calls` 自动并行）

## 框架帮你做了什么（手写 vs LangChain 1.x）

| 关注点 | 手写版（自己写） | LangChain 1.x（框架做） |
|---|---|---|
| 工具调用解析 | 自己写正则 / 解析 JSON | Function Calling 协议自动解析 |
| stop 控制 | 自己传 `stop=["Observation:"]` | Function Calling 不需要 |
| scratchpad 拼接 | 自己累积字符串 | LangGraph state messages 自动累加 |
| 工具 schema | 自己写描述字典 | `@tool` 装饰器从签名 + docstring 生成 |
| 并行调用 | 自己写 ThreadPoolExecutor | 多 `tool_calls` 自动并行 |
| 死循环防御 | 自己写 MAX_STEPS + 重复检测 | `recursion_limit` 参数 |
| 流式输出 | 自己实现 | `agent.stream(...)` 内置 |
| 可观测 | 自己 print | 接 LangSmith 自动 trace |

**结论**：框架版核心逻辑约 60 行（手写版约 130 行），省掉的全是样板代码。代价是控制粒度变粗、调试要懂 LangGraph 内部。

## 运行

```powershell
pip install -r requirements.txt
copy .env.example .env   # 填 DEEPSEEK_API_KEY
python agent.py
```

## 示例

```
You: what is the temperature difference between 深圳 and 北京?

  → 工具 get_temperature({'city': '深圳'})
  → 工具 get_temperature({'city': '北京'})    ← 注意：两个 tool_call 并行下发
  ← 工具结果: 北京 当前气温 18°C
  → 工具 calculator({'expression': '29 - 18'})
  ← 工具结果: 11
  ✓ 答案就绪

✅ Final Answer: 深圳当前气温为 29°C，北京当前气温为 18°C，两地温差为 11°C
```

注意 LangChain 1.x 默认就把 `get_temperature(深圳)` 和 `get_temperature(北京)` **并行**下发——这是手写版默认没做的优化。

## 接 LangSmith trace（可选）

`.env` 取消注释三行 `LANGCHAIN_*`，跑完去 [smith.langchain.com](https://smith.langchain.com) 看完整调用链：每步 LLM 输入输出 / token 用量 / 工具结果。比 print 调试省 80% 时间。

## 项目结构

```
langchain-react-agent/
├── agent.py            # create_agent + LangGraph CompiledStateGraph
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

## 面试讲解要点

- **LangChain 1.x 重构**：`AgentExecutor + create_react_agent` 已下线，统一到 `create_agent`（底层 LangGraph）。这是 2025 年的最新生态变化
- **Function Calling vs ReAct 文本**：1.x 走 Function Calling 协议（结构化 JSON），不再要求 LLM 输出 `Action: xxx / Action Input: xxx` 文本格式——更稳定、可并行
- **`@tool` 装饰器**：从函数签名 + docstring 自动生成 JSON Schema，省掉手写
- **为什么还要会手写**：框架出问题（解析失败、死循环、token 爆）必须懂底层原理才能 debug——见配套手写版仓库

## 许可

MIT

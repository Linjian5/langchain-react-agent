# LangChain 版 ReAct Agent

用 **LangChain `create_react_agent` + `AgentExecutor`** 实现的 ReAct Agent，作为 [react-agent-from-scratch](https://github.com/Linjian5/react-agent-from-scratch)（手写版）的**框架对照实现**——同样的 3 个工具、同样的 ReAct 范式，对比"手写 vs 框架"的代码量与控制粒度。

后端走 OpenAI 兼容协议，默认 [DeepSeek](https://platform.deepseek.com/)，改 `MODEL` + `OPENAI_BASE_URL` 即可切 OpenAI / Kimi。

## 框架帮你做了什么（手写 vs LangChain）

| 关注点 | 手写版（自己写） | LangChain 版（框架做） |
|---|---|---|
| Thought/Action 解析 | 自己写正则 | `ReActSingleInputOutputParser` |
| stop 控制 | 自己传 `stop=["Observation:"]` | `AgentExecutor` 内部处理 |
| scratchpad 拼接 | 自己累积字符串 | `agent_scratchpad` 占位符自动填 |
| 工具 schema | 自己写描述字典 | `@tool` 装饰器从签名 + docstring 生成 |
| 解析失败处理 | 自己兜底 | `handle_parsing_errors=True` |
| 死循环防御 | 自己写 MAX_STEPS + 重复检测 | `max_iterations` 参数 |
| 可观测 | 自己 print | 接 LangSmith 自动 trace |

**结论**：框架版核心逻辑约 40 行（手写版约 130 行），省掉的全是样板代码。代价是控制粒度变粗、调试要懂框架内部。

## 运行

```powershell
pip install -r requirements.txt
copy .env.example .env   # 填 DEEPSEEK_API_KEY
python agent.py
```

## 示例（同手写版）

```
You: 深圳和北京现在的温差是多少度？

> Entering new AgentExecutor chain...
Thought: 我需要先查深圳的气温
Action: get_temperature
Action Input: 深圳
Observation: 深圳 当前气温 29°C
Thought: 再查北京的气温
Action: get_temperature
Action Input: 北京
Observation: 北京 当前气温 18°C
Thought: 计算温差
Action: calculator
Action Input: 29 - 18
Observation: 11
Thought: 我现在知道最终答案了
Final Answer: 深圳(29°C)和北京(18°C)的温差是 11°C。

> Finished chain.
✅ Final Answer: 深圳(29°C)和北京(18°C)的温差是 11°C。
```

## 接 LangSmith trace（可选）

`.env` 取消注释三行 `LANGCHAIN_*`，跑完去 [smith.langchain.com](https://smith.langchain.com) 看完整调用链：每步 LLM 输入输出 / token 用量 / 工具结果。比 print 调试省 80% 时间。

## 项目结构

```
langchain-react-agent/
├── agent.py            # create_react_agent + AgentExecutor（核心 ~40 行）
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

## 面试讲解要点

- **create_react_agent vs LangGraph**：前者是经典 ReAct 单 Agent + 工具循环；复杂编排（多 Agent / 循环 / HITL / 持久化）该上 LangGraph
- **@tool 装饰器**：从函数签名 + docstring 自动生成 JSON Schema，省掉手写
- **handle_parsing_errors**：LLM 输出不符合 ReAct 格式时，框架把错误回喂让它重试，不直接崩
- **为什么还要会手写**：框架出问题时（解析失败、死循环、token 爆）必须懂底层原理才能 debug

## 许可

MIT

"""
LangChain 版 ReAct Agent —— 用 create_react_agent + AgentExecutor 实现，
对照手写版（react-agent-from-scratch）体会"框架帮你做了什么"。

同样的 3 个工具、同样的 ReAct 范式，但：
- 不用手写 Thought/Action 解析（框架的 ReActSingleInputOutputParser 做了）
- 不用手写 stop / scratchpad 拼接（AgentExecutor 管了）
- 自动接 LangSmith trace（设了环境变量就有）
"""
import os

from dotenv import load_dotenv
from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

load_dotenv()

API_KEY = os.getenv("DEEPSEEK_API_KEY") or os.getenv("OPENAI_API_KEY")
BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.deepseek.com/v1")
MODEL = os.getenv("MODEL", "deepseek-chat")
MAX_ITERATIONS = int(os.getenv("MAX_ITERATIONS", "8"))


# ============ 工具：用 @tool 装饰器，框架自动从签名 + docstring 生成 schema ============
_CITY_TEMP = {
    "北京": 18, "上海": 22, "广州": 28, "深圳": 29,
    "杭州": 23, "成都": 20, "哈尔滨": 8,
}


@tool
def calculator(expression: str) -> str:
    """计算算术表达式，输入如 '29 - 18'，仅支持数字和 + - * / ( ) ."""
    allowed = set("0123456789+-*/(). ")
    bad = [c for c in expression if c not in allowed]
    if bad:
        return f"错误：表达式含非法字符 {bad!r}"
    try:
        return str(eval(expression, {"__builtins__": {}}, {}))
    except Exception as e:
        return f"错误：计算失败 {e}"


@tool
def get_temperature(city: str) -> str:
    """查询城市当前气温（摄氏度），输入城市名如 '北京'。"""
    city = city.strip().strip("\"'")
    if city not in _CITY_TEMP:
        return f"错误：未知城市 '{city}'，已知：{', '.join(_CITY_TEMP)}"
    return f"{city} 当前气温 {_CITY_TEMP[city]}°C"


@tool
def wiki_search(query: str) -> str:
    """搜索百科知识，输入查询词如 'Apple Remote'。"""
    db = {
        "Apple Remote": "Apple Remote 最初为 Front Row 媒体中心设计。Front Row 已停止维护。",
        "Front Row": "Front Row 是 Apple 的多媒体应用，可用键盘或 Apple Remote 控制。",
        "ReAct": "ReAct 是 Yao et al. 2022 提出的 Agent 范式，结合推理与行动。",
    }
    for k, v in db.items():
        if k.lower() in query.lower():
            return v
    return f"未找到关于 '{query}' 的条目"


TOOLS = [calculator, get_temperature, wiki_search]


# ============ ReAct Prompt（LangChain hub 的标准模板，这里内联避免联网拉取） ============
REACT_PROMPT = PromptTemplate.from_template("""尽你所能回答以下问题。你可以使用这些工具：

{tools}

请严格按以下格式：

Question: 你需要回答的输入问题
Thought: 你应该思考要做什么
Action: 要采取的动作，必须是 [{tool_names}] 之一
Action Input: 动作的输入
Observation: 动作的结果
...（Thought/Action/Action Input/Observation 可以重复 N 次）
Thought: 我现在知道最终答案了
Final Answer: 对原始问题的最终回答

开始！

Question: {input}
Thought:{agent_scratchpad}""")


def build_agent() -> AgentExecutor:
    llm = ChatOpenAI(
        model=MODEL,
        api_key=API_KEY,
        base_url=BASE_URL,
        temperature=0,
    )
    agent = create_react_agent(llm, TOOLS, REACT_PROMPT)
    return AgentExecutor(
        agent=agent,
        tools=TOOLS,
        verbose=True,                       # 打印 Thought/Action/Observation
        max_iterations=MAX_ITERATIONS,      # 防死循环
        handle_parsing_errors=True,         # 解析失败自动重试，不崩
    )


def main():
    if not API_KEY:
        print("ERROR: 未设置 DEEPSEEK_API_KEY / OPENAI_API_KEY，请先 copy .env.example 到 .env")
        return
    print(f"LangChain ReAct Agent — model={MODEL} @ {BASE_URL}")
    print(f"工具：{', '.join(t.name for t in TOOLS)}")
    print("输入问题，'exit' 退出。\n")
    executor = build_agent()
    while True:
        try:
            q = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not q:
            continue
        if q.lower() in ("exit", "quit", ":q"):
            break
        result = executor.invoke({"input": q})
        print(f"\n✅ Final Answer: {result['output']}\n")


if __name__ == "__main__":
    main()

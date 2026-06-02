"""
LangChain 1.x 版 Agent —— 用 `create_agent`（底层 LangGraph）实现，
对照手写版（react-agent-from-scratch）体会"框架帮你做了什么"。

LangChain 1.x（2025 起 GA）重构了 Agent API：
- ❌ 旧：`AgentExecutor` + `create_react_agent`（0.x 时代，已下线）
- ✅ 新：`from langchain.agents import create_agent`，返回 LangGraph CompiledStateGraph

同样的 3 个工具、同样的 ReAct 范式（Thought-Action-Observation 循环），
但框架替你做了：解析、scratchpad 拼接、stop 控制、死循环防御、流式输出、checkpointing 接口。
"""
import os
import sys

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

# 强制 stdout/stderr 用 UTF-8，避免 Windows GBK 终端无法显示 ✅ / °C 等字符
try:
    sys.stdin.reconfigure(encoding="utf-8")
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except (AttributeError, OSError):
    pass

load_dotenv()

API_KEY = os.getenv("DEEPSEEK_API_KEY") or os.getenv("OPENAI_API_KEY")
BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.deepseek.com/v1")
MODEL = os.getenv("MODEL", "deepseek-chat")
MAX_ITERATIONS = int(os.getenv("MAX_ITERATIONS", "8"))


# ============ 工具：@tool 装饰器，框架自动从签名 + docstring 生成 schema ============
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

SYSTEM_PROMPT = """你是一个乐于助人的助手。
- 必要时调用工具（calculator / get_temperature / wiki_search）
- 信息收集完后直接给最终答案，不要重复调用同一工具
- 用中文回答"""


def build_agent():
    """LangChain 1.x：create_agent 返回 CompiledStateGraph（底层 LangGraph）。"""
    llm = ChatOpenAI(
        model=MODEL,
        api_key=API_KEY,
        base_url=BASE_URL,
        temperature=0,
    )
    return create_agent(
        model=llm,
        tools=TOOLS,
        system_prompt=SYSTEM_PROMPT,
    )


def run(agent, question: str) -> str:
    """跑一轮 Agent，按 LangGraph 流式打印中间步骤。"""
    print()
    final_content = ""
    # stream_mode="values" 拿到每步完整 state；"updates" 只拿增量
    for step, chunk in enumerate(
        agent.stream(
            {"messages": [HumanMessage(content=question)]},
            stream_mode="values",
            config={"recursion_limit": MAX_ITERATIONS * 2},
        ),
        1,
    ):
        msgs = chunk.get("messages", [])
        if not msgs:
            continue
        last = msgs[-1]
        kind = type(last).__name__
        if kind == "AIMessage":
            tcs = getattr(last, "tool_calls", None) or []
            if tcs:
                for tc in tcs:
                    print(f"  → 工具 {tc['name']}({tc.get('args', {})})")
            elif last.content:
                final_content = last.content
                print(f"  ✓ 答案就绪")
        elif kind == "ToolMessage":
            preview = (last.content or "")[:80]
            print(f"  ← 工具结果: {preview}")
    return final_content


def main():
    if not API_KEY:
        print("ERROR: 未设置 DEEPSEEK_API_KEY / OPENAI_API_KEY，请先 copy .env.example 到 .env")
        return
    print(f"LangChain 1.x Agent — model={MODEL} @ {BASE_URL}")
    print(f"工具：{', '.join(t.name for t in TOOLS)}")
    print("输入问题，'exit' 退出。\n")
    agent = build_agent()
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
        answer = run(agent, q)
        print(f"\n✅ Final Answer: {answer}\n")


if __name__ == "__main__":
    main()

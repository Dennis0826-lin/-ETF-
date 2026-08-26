import os
from typing import Annotated, List, TypedDict
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph, START
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
import requests
import yfinance as yf


# 1. 工具定義
@tool
def get_etf_prices(symbols: List[str]) -> str:
    """抓取台股 ETF 價格"""
    results = []
    for symbol in symbols:
        ticker_symbol = (
            f"{symbol}.TW" if not symbol.endswith(".TW") else symbol
        )
        try:
            ticker = yf.Ticker(ticker_symbol)
            info = ticker.fast_info
            last_price, prev_close = info.last_price, info.previous_close

            if last_price and prev_close:
                change = last_price - prev_close
                change_percent = (change / prev_close) * 100
                sign = "+" if change >= 0 else ""
                results.append(
                    f"• {symbol}: {last_price:.2f} 元 ({sign}{change:.2f} / {sign}{change_percent:.2f}%)"
                )
            else:
                results.append(f"• {symbol}: 查無資料")
        except Exception as e:
            results.append(f"• {symbol}: 抓取失敗 ({str(e)})")
    return "\n".join(results)


@tool
def send_telegram_message(message: str) -> str:
    """發送訊息至 Telegram"""
    bot_token = os.environ.get("TELEGRAM_BOT_TOKEN")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID")

    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    payload = {"chat_id": chat_id, "text": message}

    response = requests.post(url, json=payload)
    return (
        "Telegram 訊息發送成功！"
        if response.status_code == 200
        else f"發送失敗: {response.text}"
    )


tools = [get_etf_prices, send_telegram_message]


# 2. Agent 建立
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]


llm = ChatGoogleGenerativeAI(model="gemini-3.6-flash")
llm_with_tools = llm.bind_tools(tools)


def chatbot(state: AgentState):
    return {"messages": [llm_with_tools.invoke(state["messages"])]}


builder = StateGraph(AgentState)
builder.add_node("chatbot", chatbot)
builder.add_node("tools", ToolNode(tools))
builder.add_edge(START, "chatbot")
builder.add_conditional_edges("chatbot", tools_condition)
builder.add_edge("tools", "chatbot")

graph = builder.compile(checkpointer=MemorySaver())

# 3. 執行入口
if __name__ == "__main__":
    config = {"configurable": {"thread_id": "daily_job"}}
    user_input = "請查詢 00685L、00631L、00878、00918 的最新股價，整理成簡潔的收盤報告並發送到 Telegram！"

    events = graph.stream(
        {"messages": [("user", user_input)]}, config, stream_mode="values"
    )
    for event in events:
        pass
    print("今日排程執行完成！")

import email.mime.multipart
import email.mime.text
import os
import smtplib
from typing import Annotated, List, TypedDict
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph, START
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
import requests
import yfinance as yf


# ==========================================
# 1. 工具定義 (Tools)
# ==========================================
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

    if not bot_token or not chat_id:
        return "Telegram 設定不完整，跳過 Telegram 發送。"

    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    payload = {"chat_id": chat_id, "text": message}

    response = requests.post(url, json=payload)
    return (
        "Telegram 訊息發送成功！"
        if response.status_code == 200
        else f"Telegram 發送失敗: {response.text}"
    )


@tool
def send_email_notification(subject: str, content: str) -> str:
    """將報告或通知內容發送至指定 Email 信箱。"""
    sender_email = os.environ.get("SENDER_EMAIL")
    sender_password = os.environ.get("SENDER_PASSWORD")
    receiver_email = os.environ.get("RECEIVER_EMAIL")

    if not sender_email or not sender_password or not receiver_email:
        return "Email 設定不完整，跳過 Email 發送。"

    try:
        msg = email.mime.multipart.MIMEMultipart()
        msg["From"] = sender_email
        msg["To"] = receiver_email
        msg["Subject"] = subject

        msg.attach(email.mime.text.MIMEText(content, "plain", "utf-8"))

        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(sender_email, sender_password)
            server.send_message(msg)

        return "Email 通知發送成功！"
    except Exception as e:
        return f"Email 發送失敗: {str(e)}"


tools = [get_etf_prices, send_telegram_message, send_email_notification]


# ==========================================
# 2. Agent 建立 (LangGraph)
# ==========================================
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]


# 使用最新支援的 gemini-3.6-flash 模型
llm = ChatGoogleGenerativeAI(
    model="gemini-3.6-flash",
    max_retries=6  # 遇到 429 自動重試，給予 API 冷卻時間)
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


# ==========================================
# 3. 執行入口
# ==========================================
if __name__ == "__main__":
    config = {"configurable": {"thread_id": "daily_job"}}
    user_input = (
    "請使用 get_etf_prices 查詢 00685L、00631L、00878、00918 的最新股價，"
    "將整理好的收盤報告同時調用 send_telegram_message 與 send_email_notification 發送出去。")

    events = graph.stream(
        {"messages": [("user", user_input)]}, config, stream_mode="values"
    )
    for event in events:
        pass
    print("今日排程執行完成！")

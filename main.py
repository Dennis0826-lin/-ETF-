import datetime
import json
import os
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Annotated, List, TypedDict

import gspread
from google.oauth2.service_account import Credentials
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
import requests
import yfinance as yf


# ==========================================
# 1. 工具定義 (Tools)
# ==========================================
@tool
def get_etf_prices(symbols: List[str]) -> str:
    """抓取台股 ETF 價格與漲跌資訊"""
    results = []
    for symbol in symbols:
        ticker_symbol = (
            f"{symbol}.TW" if not symbol.endswith(".TW") else symbol
        )
        try:
            ticker = yf.Ticker(ticker_symbol)
            info = ticker.fast_info
            last_price = info.last_price
            prev_close = info.previous_close

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
def write_to_google_sheets(
    trade_date: str, symbol: str, price: float, change: str
) -> str:
    """寫入交易資料至 Google Sheets (ETF每日監控表)"""
    try:
        gcp_secret = os.environ.get("GCP_SERVICE_ACCOUNT")
        if not gcp_secret:
            return "缺少 GCP_SERVICE_ACCOUNT 環境變數，跳過 Google Sheets 寫入。"

        service_account_info = json.loads(gcp_secret)
        scopes = [
            "https://www.googleapis.com/auth/spreadsheets",
            "https://www.googleapis.com/auth/drive",
        ]
        creds = Credentials.from_service_account_info(
            service_account_info, scopes=scopes
        )
        gc = gspread.authorize(creds)

        sh = gc.open("ETF每日監控表")
        worksheet = sh.sheet1
        worksheet.append_row([trade_date, symbol, str(price), change])
        return f"✅ 已成功將 {symbol} 寫入 Google Sheets！"
    except Exception as e:
        return f"❌ Google Sheets 寫入失敗: {str(e)}"


@tool
def write_to_notion_database(
    trade_date: str, symbol: str, price: float
) -> str:
    """寫入交易資料至 Notion Database"""
    notion_token = os.environ.get("NOTION_TOKEN")
    database_id = os.environ.get("NOTION_DATABASE_ID")

    if not notion_token or not database_id:
        return "缺少 Notion 憑證，跳過 Notion 寫入。"

    headers = {
        "Authorization": f"Bearer {notion_token}",
        "Content-Type": "application/json",
        "Notion-Version": "2022-06-28",
    }

    payload = {
        "parent": {"database_id": database_id},
        "properties": {
            "Name": {"title": [{"text": {"content": symbol}}]},
            "Date": {"date": {"start": trade_date}},
            "Price": {"number": price},
        },
    }

    url = "https://api.notion.com/v1/pages"
    try:
        response = requests.post(url, headers=headers, data=json.dumps(payload))
        if response.status_code == 200:
            return f"✅ 已成功將 {symbol} 寫入 Notion Database！"
        else:
            return f"❌ Notion 寫入失敗 ({response.status_code}): {response.text}"
    except Exception as e:
        return f"❌ Notion API 請求失敗: {str(e)}"


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
    """將報告發送至指定 Email 信箱"""
    sender_email = os.environ.get("SENDER_EMAIL")
    sender_password = os.environ.get("SENDER_PASSWORD")
    receiver_email = os.environ.get("RECEIVER_EMAIL")

    if not sender_email or not sender_password or not receiver_email:
        return "Email 設定不完整，跳過 Email 發送。"

    try:
        msg = MIMEMultipart()
        msg["From"] = sender_email
        msg["To"] = receiver_email
        msg["Subject"] = subject
        msg.attach(MIMEText(content, "plain", "utf-8"))

        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(sender_email, sender_password)
            server.send_message(msg)

        return "Email 通知發送成功！"
    except Exception as e:
        return f"Email 發送失敗: {str(e)}"


# 工具清單彙整
tools = [
    get_etf_prices,
    write_to_google_sheets,
    write_to_notion_database,
    send_telegram_message,
    send_email_notification,
]


# ==========================================
# 2. Agent 建立 (LangGraph)
# ==========================================
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]


# 讀取 API Key
api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")

if not api_key:
    raise ValueError(
        "未偵測到 GEMINI_API_KEY 或 GOOGLE_API_KEY！"
        "請確認 GitHub Repository -> Settings -> Secrets and variables -> Actions 中已新增對應的 Secret。"
    )

# 設定模型為 gemini-3.6-flash，將重試次數控制為 2 次
llm = ChatGoogleGenerativeAI(
    model="gemini-3.6-flash",
    api_key=api_key,
    max_retries=2,
)
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
    today_str = datetime.date.today().strftime("%Y-%m-%d")
    
    # 加上 recursion_limit: 10 防止 Agent 進入死迴圈
    config = {
        "configurable": {"thread_id": "daily_job"},
        "recursion_limit": 10
    }

    user_input = (
        f"今天是 {today_str}。請直接執行以下任務：\n"
        "1. 使用 get_etf_prices 取得 00685L、00631L、00878、00918 的最新股價與漲跌。\n"
        "2. 將查到的每支 ETF 數據（日期、標的、價格、漲跌幅）分別寫入 write_to_google_sheets 與 write_to_notion_database。\n"
        "3. 整理完整的每日 ETF 報告，同時呼叫 send_telegram_message 與 send_email_notification 發送。"
    )

    events = graph.stream(
        {"messages": [("user", user_input)]}, config, stream_mode="values"
    )
    for event in events:
        pass
    print("今日排程執行完成！")

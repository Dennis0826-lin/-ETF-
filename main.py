# ============================================================
# main.py
# ETF 個人 AI 助理核心
#
# 特點：
# 1. 完全不使用 Checkpointer
# 2. 完全不需要 thread_id
# 3. 使用 LangGraph + Gemini Tool Calling
# 4. ETF 價格使用 yfinance history()
# 5. 保留 Google Sheets / Notion / Telegram / Email / FAISS
# ============================================================

import datetime
import json
import os
import smtplib

from tavily import TavilyClient
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Annotated, List, TypedDict

import gspread
import requests
import yfinance as yf

from dotenv import load_dotenv

from google.oauth2.service_account import Credentials

from langchain_community.vectorstores import FAISS
from langchain_core.tools import tool
from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    GoogleGenerativeAIEmbeddings,
)

from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition


# ============================================================
# 0. 載入環境變數
# ============================================================

load_dotenv()


# ============================================================
# 1. API KEY
# ============================================================

GEMINI_API_KEY = (
    os.environ.get("GEMINI_API_KEY")
    or os.environ.get("GOOGLE_API_KEY")
)

if not GEMINI_API_KEY:
    raise ValueError(
        "❌ 未偵測到 GEMINI_API_KEY 或 GOOGLE_API_KEY！\n"
        "請確認 .env 裡面有設定：\n"
        "GEMINI_API_KEY=你的API_KEY"
    )


# ============================================================
# 2. ETF 價格工具
# ============================================================

@tool
def get_etf_prices(symbols: List[str]) -> str:
    """
    查詢台股 ETF 最新交易價格、前一交易日價格、
    漲跌金額、漲跌幅。

    例如：
    get_etf_prices(["00878"])
    get_etf_prices(["00685L", "00631L", "00878", "00918"])
    """

    if not symbols:
        return "❌ 沒有提供 ETF 代號。"

    results = []

    for symbol in symbols:

        try:
            # ------------------------------------------------
            # 清理輸入
            # ------------------------------------------------

            symbol = str(symbol).strip().upper()

            if symbol.endswith(".TW"):
                ticker_symbol = symbol
                display_symbol = symbol[:-3]
            else:
                ticker_symbol = f"{symbol}.TW"
                display_symbol = symbol

            # ------------------------------------------------
            # Yahoo Finance
            # 使用 history()，不使用 fast_info
            # ------------------------------------------------

            ticker = yf.Ticker(ticker_symbol)

            df = ticker.history(
                period="5d",
                interval="1d",
                auto_adjust=False
            )

            # ------------------------------------------------
            # 沒資料
            # ------------------------------------------------

            if df is None or df.empty:
                results.append(
                    f"• {display_symbol}: "
                    f"❌ Yahoo Finance 查無資料"
                )
                continue

            # ------------------------------------------------
            # 只留下有效 Close
            # ------------------------------------------------

            df = df.dropna(subset=["Close"])

            if df.empty:
                results.append(
                    f"• {display_symbol}: "
                    f"❌ 沒有有效收盤價"
                )
                continue

            # ------------------------------------------------
            # 最新交易日
            # ------------------------------------------------

            latest_close = float(df["Close"].iloc[-1])
            latest_date = df.index[-1]

            try:
                latest_date_str = latest_date.strftime("%Y-%m-%d")
            except Exception:
                latest_date_str = str(latest_date)

            # ------------------------------------------------
            # 前一交易日
            # ------------------------------------------------

            if len(df) >= 2:

                previous_close = float(df["Close"].iloc[-2])

                change = latest_close - previous_close

                if previous_close != 0:
                    change_percent = (
                        change / previous_close * 100
                    )
                else:
                    change_percent = 0

                sign = "+" if change >= 0 else ""

                results.append(
                    f"• {display_symbol}: "
                    f"{latest_close:.2f} 元 "
                    f"({sign}{change:.2f} / "
                    f"{sign}{change_percent:.2f}%) "
                    f"[最新交易日：{latest_date_str}]"
                )

            else:

                results.append(
                    f"• {display_symbol}: "
                    f"{latest_close:.2f} 元 "
                    f"[最新交易日：{latest_date_str}]"
                )

        except Exception as e:

            results.append(
                f"• {display_symbol}: "
                f"❌ 抓取失敗：{type(e).__name__}: {str(e)}"
            )

    return "\n".join(results)

# ============================================================
# 3. Web Search - Tavily
# ============================================================

@tool
def search_web(query: str) -> str:
    """
    使用 Tavily 搜尋最新網路資訊。
    適合查詢最新股市新聞、ETF消息、公司公告、
    技術文件及其他需要即時網路資訊的問題。
    """

    tavily_api_key = os.environ.get(
        "TAVILY_API_KEY"
    )

    if not tavily_api_key:
        return (
            "❌ 找不到 TAVILY_API_KEY。\n"
            "請確認 .env 已設定：\n"
            "TAVILY_API_KEY=你的API_KEY"
        )

    try:

        client = TavilyClient(
            api_key=tavily_api_key
        )

        response = client.search(
            query=query,
            search_depth="advanced",
            max_results=5
        )

        results = response.get(
            "results",
            []
        )

        if not results:
            return (
                f"⚠️ Tavily 找不到與「{query}」"
                "相關的網路資料。"
            )

        output = []

        for i, result in enumerate(
            results,
            start=1
        ):

            title = result.get(
                "title",
                "無標題"
            )

            url = result.get(
                "url",
                ""
            )

            content = result.get(
                "content",
                ""
            )

            output.append(
                f"[{i}] {title}\n"
                f"URL: {url}\n"
                f"{content}"
            )

        return "\n\n".join(output)

    except Exception as e:

        return (
            f"❌ Tavily 搜尋失敗："
            f"{type(e).__name__}: {str(e)}"
        )
        
# ============================================================
# 3. Google Sheets
# ============================================================

@tool
def write_to_google_sheets(
    trade_date: str,    
    symbol: str,
    price: float,
    change: str
) -> str:
    """
    將 ETF 每日交易資料寫入 Google Sheets。
    """

    try:

        gcp_secret = os.environ.get(
            "GCP_SERVICE_ACCOUNT"
        )

        if not gcp_secret:

            return (
                "⚠️ 缺少 GCP_SERVICE_ACCOUNT 環境變數，"
                "跳過 Google Sheets 寫入。"
            )

        service_account_info = json.loads(
            gcp_secret
        )

        scopes = [
            "https://www.googleapis.com/auth/spreadsheets",
            "https://www.googleapis.com/auth/drive",
        ]

        creds = Credentials.from_service_account_info(
            service_account_info,
            scopes=scopes
        )

        gc = gspread.authorize(creds)

        sh = gc.open("ETF每日監控表")

        worksheet = sh.sheet1

        worksheet.append_row(
            [
                trade_date,
                symbol,
                str(price),
                change,
            ]
        )

        return (
            f"✅ 已成功將 {symbol} "
            f"寫入 Google Sheets！"
        )

    except Exception as e:

        return (
            f"❌ Google Sheets 寫入失敗："
            f"{type(e).__name__}: {str(e)}"
        )


# ============================================================
# 4. Notion
# ============================================================

@tool
def write_to_notion_database(
    trade_date: str,
    symbol: str,
    price: float
) -> str:
    """
    將 ETF 資料寫入 Notion Database。
    """

    notion_token = os.environ.get(
        "NOTION_TOKEN"
    )

    database_id = os.environ.get(
        "NOTION_DATABASE_ID"
    )

    if not notion_token or not database_id:

        return (
            "⚠️ 缺少 Notion 憑證，"
            "跳過 Notion 寫入。"
        )

    headers = {
        "Authorization": f"Bearer {notion_token}",
        "Content-Type": "application/json",
        "Notion-Version": "2022-06-28",
    }

    payload = {
        "parent": {
            "database_id": database_id
        },
        "properties": {

            "Name": {
                "title": [
                    {
                        "text": {
                            "content": symbol
                        }
                    }
                ]
            },

            "Date": {
                "date": {
                    "start": trade_date
                }
            },

            "Price": {
                "number": price
            },
        },
    }

    url = "https://api.notion.com/v1/pages"

    try:

        response = requests.post(
            url,
            headers=headers,
            json=payload,
            timeout=30
        )

        if response.status_code in (200, 201):

            return (
                f"✅ 已成功將 {symbol} "
                f"寫入 Notion Database！"
            )

        return (
            f"❌ Notion 寫入失敗 "
            f"({response.status_code})："
            f"{response.text}"
        )

    except Exception as e:

        return (
            f"❌ Notion API 請求失敗："
            f"{type(e).__name__}: {str(e)}"
        )


# ============================================================
# 5. Telegram
# ============================================================

@tool
def send_telegram_message(
    message: str
) -> str:
    """
    發送訊息至 Telegram。
    """

    bot_token = os.environ.get(
        "TELEGRAM_BOT_TOKEN"
    )

    chat_id = os.environ.get(
        "TELEGRAM_CHAT_ID"
    )

    if not bot_token or not chat_id:

        return (
            "⚠️ Telegram 設定不完整，"
            "跳過 Telegram 發送。"
        )

    url = (
        f"https://api.telegram.org/"
        f"bot{bot_token}/sendMessage"
    )

    payload = {
        "chat_id": chat_id,
        "text": message,
    }

    try:

        response = requests.post(
            url,
            json=payload,
            timeout=30
        )

        if response.status_code == 200:

            return "✅ Telegram 訊息發送成功！"

        return (
            f"❌ Telegram 發送失敗："
            f"{response.text}"
        )

    except Exception as e:

        return (
            f"❌ Telegram API 錯誤："
            f"{type(e).__name__}: {str(e)}"
        )


# ============================================================
# 6. Email
# ============================================================

@tool
def send_email_notification(
    subject: str,
    content: str
) -> str:
    """
    將 ETF 報告發送至指定 Email。
    """

    sender_email = os.environ.get(
        "SENDER_EMAIL"
    )

    sender_password = os.environ.get(
        "SENDER_PASSWORD"
    )

    receiver_email = os.environ.get(
        "RECEIVER_EMAIL"
    )

    if (
        not sender_email
        or not sender_password
        or not receiver_email
    ):

        return (
            "⚠️ Email 設定不完整，"
            "跳過 Email 發送。"
        )

    try:

        msg = MIMEMultipart()

        msg["From"] = sender_email
        msg["To"] = receiver_email
        msg["Subject"] = subject

        msg.attach(
            MIMEText(
                content,
                "plain",
                "utf-8"
            )
        )

        with smtplib.SMTP_SSL(
            "smtp.gmail.com",
            465,
            timeout=30
        ) as server:

            server.login(
                sender_email,
                sender_password
            )

            server.send_message(msg)

        return "✅ Email 通知發送成功！"

    except Exception as e:

        return (
            f"❌ Email 發送失敗："
            f"{type(e).__name__}: {str(e)}"
        )


# ============================================================
# 7. 個人知識庫 FAISS
# ============================================================

@tool
def search_personal_docs(
    query: str
) -> str:
    """
    查詢個人專屬知識庫。
    """

    try:

        index_path = "faiss_index"

        if not os.path.exists(index_path):

            return (
                "⚠️ 目前尚未建立個人知識庫索引 "
                "(faiss_index 資料夾不存在)。"
            )

        api_key = (
            os.environ.get("GEMINI_API_KEY")
            or os.environ.get("GOOGLE_API_KEY")
        )

        if not api_key:

            return (
                "❌ 找不到 Gemini API Key。"
            )

        embeddings = GoogleGenerativeAIEmbeddings(
            model="text-embedding-004",
            google_api_key=api_key
        )

        vectorstore = FAISS.load_local(
            index_path,
            embeddings,
            allow_dangerous_deserialization=True
        )

        docs = vectorstore.similarity_search(
            query,
            k=3
        )

        if not docs:

            return "查無相關個人文件紀錄。"

        results = []

        for d in docs:

            source = d.metadata.get(
                "source",
                "未知"
            )

            results.append(
                f"• [文件來源：{source}]\n"
                f"{d.page_content}"
            )

        return "\n\n".join(results)

    except Exception as e:

        return (
            f"❌ 知識庫查詢失敗："
            f"{type(e).__name__}: {str(e)}"
        )


# ============================================================
# 8. Tools 集合
# ============================================================

tools = [
    get_etf_prices,
    search_web,
    write_to_google_sheets,
    write_to_notion_database,
    send_telegram_message,
    send_email_notification,
    search_personal_docs,
]


# ============================================================
# 9. Agent State
# ============================================================

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]


# ============================================================
# 10. Gemini
#
# 注意：
# 完全不使用 Checkpointer
# ============================================================

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=GEMINI_API_KEY,
    temperature=0,
)

llm_with_tools = llm.bind_tools(tools)


# ============================================================
# 11. Chatbot
# ============================================================

def chatbot(state: AgentState):

    response = llm_with_tools.invoke(
        state["messages"]
    )

    return {
        "messages": [response]
    }


# ============================================================
# 12. 建立 LangGraph
#
# ★★★ 重要 ★★★
#
# 這裡完全沒有：
#
# checkpointer=
# MemorySaver
# SqliteSaver
# PostgresSaver
# thread_id
#
# ============================================================

builder = StateGraph(AgentState)

builder.add_node(
    "chatbot",
    chatbot
)

builder.add_node(
    "tools",
    ToolNode(tools)
)

builder.add_edge(
    START,
    "chatbot"
)

builder.add_conditional_edges(
    "chatbot",
    tools_condition
)

builder.add_edge(
    "tools",
    "chatbot"
)


# ============================================================
# 13. Compile
#
# ★★★ 完全不帶 Checkpointer ★★★
# ============================================================

graph = builder.compile()


# ============================================================
# 14. Debug 資訊
# ============================================================

def get_debug_info():

    checkpointer = getattr(
        graph,
        "checkpointer",
        None
    )

    return {
        "main_file": os.path.abspath(__file__),
        "checkpointer": str(checkpointer),
        "checkpointer_is_none": checkpointer is None,
        "tools": [
            t.name for t in tools
        ],
    }


# ============================================================
# 15. 直接執行 main.py 時的測試
#
# python main.py
#
# 不會啟動 Streamlit
# 只測試 ETF 查價工具
# ============================================================
if __name__ == "__main__":

    print("=" * 70)

    print(
        "ETF AI Assistant - main.py 測試"
    )

    print("=" * 70)

    print()

    print("目前 main.py：")
    print(
        os.path.abspath(__file__)
    )

    print()

    print("Graph Checkpointer：")
    print(
        getattr(
            graph,
            "checkpointer",
            None
        )
    )

    print()

    print("開始測試 00878 ETF 查價...")
    print()

    try:

        result =get_etf_prices.invoke({
    "symbols": ["00878"]
})

        print(result)

    except Exception as e:

        print(
            "❌ ETF 查價測試失敗："
        )

        print(
            type(e).__name__,
            str(e)
        )

    print()

    print("=" * 70)

    print("測試結束")

    print("=" * 70)

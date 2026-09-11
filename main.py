```python
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
# 6. Telegram 支援固定格式每日監控報告
# ============================================================


# ============================================================
# 0. Imports
# ============================================================

import datetime
import json
import os
import smtplib

from typing import Annotated, List, TypedDict

import gspread
import requests
import yfinance as yf

from dotenv import load_dotenv

from .mime.multipart import MIMEMultipart
from .mime.text import MIMEText

from google.oauth2.service_account import Credentials

from tavily import TAVILYClient

from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from langgraph.graph import START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition


# ============================================================
# 1. 載入環境變數
# ============================================================

load_dotenv()


# ============================================================
# 2. Gemini API KEY
# ============================================================

GEMINI_API_KEY = (
    os.environ.get("GEMINI_API_KEY")
    or os.environ.get("GOOGLE_API_KEY")
)

if not GEMINI_API_KEY:
    raise ValueError(
        "❌ 未偵測到 GEMINI_API_KEY 或 GOOGLE_API_KEY！\n"
        "請確認環境變數已設定。"
    )


# ============================================================
# 3. ETF 價格工具
# ============================================================

@tool
def get_etf_prices(symbols: List[str]) -> str:
    """
    查詢台股 ETF 最新交易價格、前一交易日價格、
    漲跌金額、漲跌幅與最新交易日。

    例如：

    get_etf_prices(["00878"])

    get_etf_prices([
        "00685L",
        "00631L",
        "00878",
        "00918"
    ])
    """

    if not symbols:
        return "❌ 沒有提供 ETF 代號。"

    results = []

    for symbol in symbols:

        display_symbol = str(symbol).strip().upper()

        try:

            # ------------------------------------------------
            # 清理 ETF 代號
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
            # 使用 history()
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

            df = df.dropna(
                subset=["Close"]
            )

            if df.empty:

                results.append(
                    f"• {display_symbol}: "
                    f"❌ 沒有有效收盤價"
                )

                continue

            # ------------------------------------------------
            # 最新交易日
            # ------------------------------------------------

            latest_close = float(
                df["Close"].iloc[-1]
            )

            latest_date = df.index[-1]

            try:

                latest_date_str = (
                    latest_date.strftime(
                        "%Y-%m-%d"
                    )
                )

            except Exception:

                latest_date_str = str(
                    latest_date
                )

            # ------------------------------------------------
            # 前一交易日
            # ------------------------------------------------

            if len(df) >= 2:

                previous_close = float(
                    df["Close"].iloc[-2]
                )

                change = (
                    latest_close
                    - previous_close
                )

                if previous_close != 0:

                    change_percent = (
                        change
                        / previous_close
                        * 100
                    )

                else:

                    change_percent = 0.0

                sign = (
                    "+"
                    if change >= 0
                    else ""
                )

                results.append(
                    f"• {display_symbol}: "
                    f"{latest_close:.2f} 元 "
                    f"({sign}{change:.2f} / "
                    f"{sign}{change_percent:.2f}%) "
                    f"[最新交易日："
                    f"{latest_date_str}]"
                )

            else:

                results.append(
                    f"• {display_symbol}: "
                    f"{latest_close:.2f} 元 "
                    f"[最新交易日："
                    f"{latest_date_str}]"
                )

        except Exception as e:

            results.append(
                f"• {display_symbol}: "
                f"❌ 抓取失敗："
                f"{type(e).__name__}: {str(e)}"
            )

    return "\n".join(results)


# ============================================================
# 4. ETF 投資組合計算工具
# ============================================================

@tool
def calculate_portfolio(
    holdings: List[dict]
) -> str:
    """
    根據 ETF 持有張數、平均成本與最新價格，
    計算投入成本、目前市值、未實現損益與報酬率。

    holdings 格式：

    [
        {
            "symbol": "00918",
            "shares": 200,
            "avg_cost": 22.70,
            "current_price": 35.38
        }
    ]
    """

    if not holdings:

        return (
            "❌ 沒有提供投資組合資料。"
        )

    results = []

    total_cost = 0.0
    total_market_value = 0.0

    for item in holdings:

        try:

            symbol = str(
                item["symbol"]
            ).strip().upper()

            shares = float(
                item["shares"]
            )

            avg_cost = float(
                item["avg_cost"]
            )

            current_price = float(
                item["current_price"]
            )

            # ------------------------------------------------
            # ETF：
            # 1 張 = 1,000 股
            # ------------------------------------------------

            cost = (
                shares
                * 1000
                * avg_cost
            )

            market_value = (
                shares
                * 1000
                * current_price
            )

            profit = (
                market_value
                - cost
            )

            if cost != 0:

                return_rate = (
                    profit
                    / cost
                    * 100
                )

            else:

                return_rate = 0.0

            sign = (
                "+"
                if profit >= 0
                else ""
            )

            results.append(
                f"• {symbol}: "
                f"持有 {shares:.0f} 張，"
                f"投入成本 {cost:,.0f} 元，"
                f"目前市值 {market_value:,.0f} 元，"
                f"未實現損益 "
                f"{sign}{profit:,.0f} 元，"
                f"報酬率 "
                f"{return_rate:+.2f}%"
            )

            total_cost += cost
            total_market_value += (
                market_value
            )

        except Exception as e:

            results.append(
                f"• {item.get('symbol', 'UNKNOWN')}: "
                f"❌ 計算失敗："
                f"{type(e).__name__}: {str(e)}"
            )

    # ------------------------------------------------
    # 投資組合總計
    # ------------------------------------------------

    total_profit = (
        total_market_value
        - total_cost
    )

    if total_cost != 0:

        total_return_rate = (
            total_profit
            / total_cost
            * 100
        )

    else:

        total_return_rate = 0.0

    sign = (
        "+"
        if total_profit >= 0
        else ""
    )

    results.append("")

    results.append(
        f"📊 投資組合總計："
        f"投入成本 "
        f"{total_cost:,.0f} 元，"
        f"目前市值 "
        f"{total_market_value:,.0f} 元，"
        f"未實現損益 "
        f"{sign}{total_profit:,.0f} 元，"
        f"總報酬率 "
        f"{total_return_rate:+.2f}%"
    )

    return "\n".join(results)


# ============================================================
# 5. Web Search
# ============================================================

@tool
def search_web(
    query: str
) -> str:
    """
    使用 Tavily 搜尋最新網路資訊。

    適合查詢：
    - 最新股市新聞
    - ETF 消息
    - 公司公告
    - 技術文件
    - 其他需要即時網路資訊的問題
    """

    tavily_api_key = os.environ.get(
        "TAVILY_API_KEY"
    )

    if not tavily_api_key:

        return (
            "❌ 找不到 TAVILY_API_KEY。\n"
            "請確認環境變數已設定。"
        )

    try:

        client = TAVILYClient(
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
                f"⚠️ TAVILY 找不到與"
                f"「{query}」相關的網路資料。"
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
            f"❌ TAVILY 搜尋失敗："
            f"{type(e).__name__}: {str(e)}"
        )


# ============================================================
# 6. Google Sheets
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
                "⚠️ 缺少 GCP_SERVICE_ACCOUNT "
                "環境變數，"
                "跳過 Google Sheets 寫入。"
            )

        service_account_info = json.loads(
            gcp_secret
        )

        scopes = [
            "https://www.googleapis.com/auth/spreadsheets",
            "https://www.googleapis.com/auth/drive",
        ]

        creds = (
            Credentials
            .from_service_account_info(
                service_account_info,
                scopes=scopes
            )
        )

        gc = gspread.authorize(
            creds
        )

        sh = gc.open(
            "ETF每日監控表"
        )

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
# 7. Notion
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

    if (
        not notion_token
        or not database_id
    ):

        return (
            "⚠️ 缺少 Notion 憑證，"
            "跳過 Notion 寫入。"
        )

    headers = {
        "Authorization": (
            f"Bearer {notion_token}"
        ),
        "Content-Type": (
            "application/json"
        ),
        "Notion-Version": (
            "2022-06-28"
        ),
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

    url = (
        "https://api.notion.com/v1/pages"
    )

    try:

        response = requests.post(
            url,
            headers=headers,
            json=payload,
            timeout=30
        )

        if response.status_code in (
            200,
            201
        ):

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
# 8. Telegram
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

    if (
        not bot_token
        or not chat_id
    ):

        return (
            "⚠️ Telegram 設定不完整，"
            "跳過 Telegram 發送。"
        )

    url = (
        "https://api.telegram.org/"
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

            return (
                "✅ Telegram 訊息發送成功！"
            )

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
# Telegram 報告格式工具
# ============================================================

@tool
def build_telegram_report(
    trade_date: str,
    market_summary: str,
    portfolio_summary: str
) -> str:
    """
    建立固定格式的 ETF Telegram 每日監控報告。
    """

    report = f"""📊 ETF 每日監控
{trade_date}

━━━━━━━━━━━━━━
📈 今日行情
━━━━━━━━━━━━━━

{market_summary}

━━━━━━━━━━━━━━
💰 投資組合
━━━━━━━━━━━━━━

{portfolio_summary}

━━━━━━━━━━━━━━
📋 Google Sheets
━━━━━━━━━━━━━━

✅ 今日資料已更新
"""

    return report


# ============================================================
# 10. Email
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

            server.send_message(
                msg
            )

        return (
            "✅ Email 通知發送成功！"
        )

    except Exception as e:

        return (
            f"❌ Email 發送失敗："
            f"{type(e).__name__}: {str(e)}"
        )


# ============================================================
# 11. 個人知識庫 FAISS
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

        if not os.path.exists(
            index_path
        ):

            return (
                "⚠️ 目前尚未建立個人知識庫索引 "
                "(faiss_index 資料夾不存在)。"
            )

        api_key = (
            os.environ.get(
                "GEMINI_API_KEY"
            )
            or os.environ.get(
                "GOOGLE_API_KEY"
            )
        )

        if not api_key:

            return (
                "❌ 找不到 Gemini API Key。"
            )

        embeddings = (
            HuggingFaceEmbeddings(
                model_name=(
                    "all-MiniLM-L6-v2"
                )
            )
        )

        vectorstore = FAISS.load_local(
            index_path,
            embeddings,
            allow_dangerous_deserialization=True
        )

        docs = (
            vectorstore
            .similarity_search(
                query,
                k=3
            )
        )

        if not docs:

            return (
                "查無相關個人文件紀錄。"
            )

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

        return "\n\n".join(
            results
        )

    except Exception as e:

        return (
            f"❌ 知識庫查詢失敗："
            f"{type(e).__name__}: {str(e)}"
        )


# ============================================================
# 12. Tools 集合
# ============================================================

tools = [
    get_etf_prices,
    calculate_portfolio,
    build_telegram_report,
    search_web,
    write_to_google_sheets,
    write_to_notion_database,
    send_telegram_message,
    send_email_notification,
    search_personal_docs,
]


# ============================================================
# 13. Agent State
# ============================================================

class AgentState(TypedDict):

    messages: Annotated[
        list,
        add_messages
    ]


# ============================================================
# 14. Gemini
#
# 完全不使用 Checkpointer
# ============================================================

llm = ChatGoogleGenerativeAI(
    model="gemini-3.6-flash",
    google_api_key=GEMINI_API_KEY,
)

llm_with_tools = llm.bind_tools(
    tools
)


# ============================================================
# 15. Chatbot
# ============================================================

def chatbot(
    state: AgentState
):

    response = (
        llm_with_tools.invoke(
            state["messages"]
        )
    )

    return {
        "messages": [
            response
        ]
    }


# ============================================================
# 16. 建立 LangGraph
#
# 完全沒有：
#
# checkpointer=
# MemorySaver
# SqliteSaver
# PostgresSaver
# thread_id
# ============================================================

builder = StateGraph(
    AgentState
)

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
# 17. Compile
#
# 完全不帶 Checkpointer
# ============================================================

graph = builder.compile()


# ============================================================
# 18. Debug 資訊
# ============================================================

def get_debug_info():

    checkpointer = getattr(
        graph,
        "checkpointer",
        None
    )

    return {
        "main_file": (
            os.path.abspath(__file__)
        ),
        "checkpointer": str(
            checkpointer
        ),
        "checkpointer_is_none": (
            checkpointer is None
        ),
        "tools": [
            t.name
            for t in tools
        ],
    }


# ============================================================
# 19. 直接執行 main.py 時的測試
#
# python main.py
#
# 不會啟動 Streamlit
# 只執行基本測試
# ============================================================

if __name__ == "__main__":

    print("=" * 70)

    print(
        "ETF AI Assistant - main.py 測試"
    )

    print("=" * 70)

    print()

    # --------------------------------------------------------
    # 顯示目前 main.py
    # --------------------------------------------------------

    print("目前 main.py：")

    print(
        os.path.abspath(__file__)
    )

    print()

    # --------------------------------------------------------
    # Graph Checkpointer
    # --------------------------------------------------------

    print(
        "Graph Checkpointer："
    )

    print(
        getattr(
            graph,
            "checkpointer",
            None
        )
    )

    print()

    # --------------------------------------------------------
    # Tool 清單
    # --------------------------------------------------------

    print("目前 Tools：")

    for tool_item in tools:

        print(
            f"  - {tool_item.name}"
        )

    print()

    # --------------------------------------------------------
    # Test 1：ETF 查價
    # --------------------------------------------------------

    print(
        "開始測試 00878 ETF 查價..."
    )

    print()

    try:

        result = (
            get_etf_prices.invoke(
                {
                    "symbols": [
                        "00878"
                    ]
                }
            )
        )

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

    # --------------------------------------------------------
    # Test 2：Telegram Report Tool
    # --------------------------------------------------------

    print(
        "開始測試 Telegram 報告格式..."
    )

    print()

    try:

        report = (
            build_telegram_report.invoke(
                {
                    "trade_date": (
                        datetime.date.today()
                        .strftime("%Y-%m-%d")
                    ),
                    "market_summary": (
                        "00918   35.38  🔻 -0.17%\n"
                        "00878   34.15  🔺 +0.23%\n"
                        "00631L  37.15  🔻 -0.91%\n"
                        "00685L  12.26  🔻 -0.73%"
                    ),
                    "portfolio_summary": (
                        "投入成本\n"
                        "$9,328,950\n\n"
                        "目前市值\n"
                        "$13,958,450\n\n"
                        "未實現損益\n"
                        "+$4,629,500\n\n"
                        "總報酬率\n"
                        "+49.63%"
                    ),
                }
            )
        )

        print(report)

    except Exception as e:

        print(
            "❌ Telegram 報告格式測試失敗："
        )

        print(
            type(e).__name__,
            str(e)
        )

    print()

    # --------------------------------------------------------
    # Test 3：Gemini + LangGraph
    # --------------------------------------------------------

    print("=" * 70)

    print(
        "開始測試 Gemini 3.6 Flash "
        "+ LangGraph + Tool Calling..."
    )

    print("=" * 70)

    print()

    try:

        result = graph.invoke(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": (
                            "請查詢我的個人 ETF 知識庫，"
                            "告訴我目前 00878、00918、"
                            "00631L、00685L "
                            "各持有幾張，以及各自的平均成本。"
                        )
                    }
                ]
            }
        )

        print(
            "Gemini + LangGraph 最終回應："
        )

        print()

        for message in result["messages"]:

            print(
                f"[{type(message).__name__}]"
            )

            print(
                message.content
            )

            print()

    except Exception as e:

        print(
            "❌ Gemini + LangGraph 測試失敗："
        )

        print(
            type(e).__name__,
            str(e)
        )

    print()

    print("=" * 70)

    print(
        "測試結束"
    )

    print("=" * 70)
```

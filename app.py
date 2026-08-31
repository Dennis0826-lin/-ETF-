import streamlit as st
from main import graph  # 匯入 main.py 中定義好的 LangGraph Agent

# 頁面基本設定
st.set_page_config(page_title="ETF & 個人助理", page_icon="🤖", layout="centered")
st.title("🤖 我的個人 AI 助理")


# 使用 cache_resource 包裝，確保模組資源正確載入
@st.cache_resource
def get_agent_runner():
    return graph


agent = get_agent_runner()

# 初始化對話紀錄與 Session Thread ID
if "messages" not in st.session_state:
    st.session_state.messages = []

if "thread_id" not in st.session_state:
    st.session_state.thread_id = "streamlit_session"

# 渲染歷史對話訊息
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# 處理使用者輸入
if prompt := st.chat_input(
    "請輸入你的問題（例如：查詢 00878 最新股價、檢索個人文件）..."
):
    # 1. 顯示並紀錄使用者訊息
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2. 呼叫 LangGraph Agent
    with st.chat_message("assistant"):
        with st.spinner("思考中並處理工具呼叫..."):
            try:
                # 建立包含 thread_id 的 config 字典 (與圖片邏輯一致)
                inputs = {"messages": [("user", prompt)]}
                config = {
                    "configurable": {"thread_id": st.session_state.thread_id}
                }

                # 帶入 config 參數執行 invoke
                response = agent.invoke(inputs, config=config)

                # 提取最終回答內容
                if isinstance(response, dict) and "messages" in response:
                    reply = response["messages"][-1].content
                else:
                    reply = str(response)

            except Exception as e:
                reply = f"執行出錯：{str(e)}"

        # 顯示 AI 回應並寫入紀錄
        st.markdown(reply)
        st.session_state.messages.append(
            {"role": "assistant", "content": reply}
        )

import streamlit as st
from main import graph  # 匯入 main.py 中定義好的 LangGraph Agent

# 頁面基本設定
st.set_page_config(page_title="ETF & 個人助理", page_icon="🤖", layout="centered")
st.title("🤖 我的個人 AI 助理")

# 初始化對話紀錄與固定 Session Thread ID
if "messages" not in st.session_state:
    st.session_state.messages = []

if "thread_id" not in st.session_state:
    st.session_state.thread_id = "streamlit_default_user"

# 渲染歷史對話訊息
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# 處理使用者輸入
if prompt := st.chat_input("請輸入你的問題（例如：查詢 00878 最新股價、檢索個人文件）..."):
    # 1. 顯示並紀錄使用者訊息
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2. 呼叫 LangGraph Agent
    with st.chat_message("assistant"):
        with st.spinner("思考中並處理工具呼叫..."):
            try:
                # 構建符合 LangGraph 規範的輸入與 Config
                inputs = {"messages": [("user", prompt)]}
                config = {
                    "configurable": {
                        "thread_id": st.session_state.thread_id
                    }
                }
                
                # 執行 Agent 邏輯
                response = graph.invoke(inputs, config=config)
                
                # 提取最終回答內容
                if isinstance(response, dict) and "messages" in response:
                    reply = response["messages"][-1].content
                else:
                    reply = str(response)

            except Exception as e:
                reply = f"執行出錯：{str(e)}"

        # 顯示 AI 回應
        st.markdown(reply)
        # 紀錄 AI 回應至 session
        st.session_state.messages.append({"role": "assistant", "content": reply})

import streamlit as st
from main import graph

st.set_page_config(page_title="ETF & 個人助理", page_icon="🤖", layout="centered")
st.title("🤖 我的個人 AI 助理")

if "messages" not in st.session_state:
    st.session_state.messages = []

if "thread_id" not in st.session_state:
    st.session_state.thread_id = "streamlit_user_session"

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("請輸入你的問題..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("思考中..."):
            try:
                # 必須嚴格遵循此結構傳遞 config
                inputs = {"messages": [("user", prompt)]}
                config = {
                    "configurable": {
                        "thread_id": st.session_state.thread_id
                    }
                }

                response = graph.invoke(inputs, config=config)

                if isinstance(response, dict) and "messages" in response:
                    reply = response["messages"][-1].content
                else:
                    reply = str(response)

            except Exception as e:
                reply = f"執行出錯：{str(e)}"

        st.markdown(reply)
        st.session_state.messages.append({"role": "assistant", "content": reply})

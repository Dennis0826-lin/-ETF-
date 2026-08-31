# ============================================================
# app.py
# Streamlit 個人 AI 助理
#
# 特點：
# 1. 完全不使用 Checkpointer
# 2. 不需要 thread_id
# 3. 使用 Streamlit session_state 保存對話
# 4. 將完整對話歷史傳給 LangGraph
# 5. 顯示 Debug 資訊確認實際載入的 main.py
# ============================================================

import streamlit as st
import main


# ============================================================
# 1. Streamlit 頁面設定
# ============================================================

st.set_page_config(
    page_title="ETF & 個人助理",
    page_icon="🤖",
    layout="centered",
)


# ============================================================
# 2. 標題
# ============================================================

st.title("🤖 我的個人 AI 助理")


st.caption(
    "Gemini + LangGraph + ETF Tools"
)


# ============================================================
# 3. 載入 Graph
# ============================================================

graph = main.graph


# ============================================================
# 4. DEBUG 區域
#
# 第一次測試非常重要
# ============================================================

with st.sidebar:

    st.header("🔧 系統狀態")

    st.write("目前載入的 main.py：")

    st.code(
        main.__file__
    )

    st.write("Graph Checkpointer：")

    checkpointer = getattr(
        graph,
        "checkpointer",
        None
    )

    st.code(
        str(checkpointer)
    )

    if checkpointer is None:

        st.success(
            "✅ Checkpointer = None"
        )

    else:

        st.error(
            "❌ 仍然存在 Checkpointer"
        )

    st.write("可用 Tools：")

    for tool in main.tools:

        st.write(
            f"• {tool.name}"
        )


# ============================================================
# 5. 初始化 Session State
# ============================================================

if "messages" not in st.session_state:

    st.session_state.messages = []


# ============================================================
# 6. 顯示歷史對話
# ============================================================

for message in st.session_state.messages:

    with st.chat_message(
        message["role"]
    ):

        st.markdown(
            message["content"]
        )


# ============================================================
# 7. 使用者輸入
# ============================================================

prompt = st.chat_input(
    "請輸入你的問題，例如：查詢 00878 最新股價"
)


if prompt:

    # --------------------------------------------------------
    # 顯示 User 訊息
    # --------------------------------------------------------

    st.session_state.messages.append(
        {
            "role": "user",
            "content": prompt,
        }
    )

    with st.chat_message("user"):

        st.markdown(prompt)


    # --------------------------------------------------------
    # AI 回覆
    # --------------------------------------------------------

    with st.chat_message("assistant"):

        with st.spinner(
            "AI 思考中，正在處理工具..."
        ):

            try:

                # ====================================================
                # 將完整對話歷史傳給 LangGraph
                # ====================================================

                graph_messages = []

                for msg in (
                    st.session_state.messages
                ):

                    graph_messages.append(
                        (
                            msg["role"],
                            msg["content"],
                        )
                    )


                # ====================================================
                # 呼叫 LangGraph
                #
                # ★★★ 注意 ★★★
                #
                # 沒有 config
                # 沒有 thread_id
                # 沒有 checkpoint
                #
                # ====================================================

                response = graph.invoke(
                    {
                        "messages":
                        graph_messages
                    }
                )


                # ====================================================
                # 取得最後一則 AI 訊息
                # ====================================================

                if (
                    isinstance(
                        response,
                        dict
                    )
                    and "messages"
                    in response
                    and response["messages"]
                ):

                    final_message = (
                        response["messages"][-1]
                    )

                    reply = (
                        final_message.content
                    )

                else:

                    reply = str(
                        response
                    )


                # ====================================================
                # Gemini 某些版本可能回傳 list
                # ====================================================

                if isinstance(
                    reply,
                    list
                ):

                    formatted_parts = []

                    for item in reply:

                        if isinstance(
                            item,
                            dict
                        ):

                            if "text" in item:

                                formatted_parts.append(
                                    str(
                                        item["text"]
                                    )
                                )

                            else:

                                formatted_parts.append(
                                    str(item)
                                )

                        else:

                            formatted_parts.append(
                                str(item)
                            )

                    reply = "\n".join(
                        formatted_parts
                    )


                # ====================================================
                # 空白回覆保護
                # ====================================================

                if not reply:

                    reply = (
                        "⚠️ AI 沒有回傳文字內容。"
                    )


            except Exception as e:

                reply = (
                    "❌ 執行出錯\n\n"
                    f"錯誤類型："
                    f"{type(e).__name__}\n\n"
                    f"錯誤內容："
                    f"{str(e)}"
                )


        # --------------------------------------------------------
        # 顯示 AI 回覆
        # --------------------------------------------------------

        st.markdown(
            reply
        )


        # --------------------------------------------------------
        # 儲存 AI 回覆
        # --------------------------------------------------------

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": reply,
            }
        )

from agents import Agent, Runner


agent = Agent(
    name="ETF Assistant",
    instructions="""
你是一個個人 ETF AI 助理。

你的主要任務是：
1. 協助使用者理解 ETF
2. 分析投資組合
3. 協助整理市場資訊
4. 未來可以透過工具取得即時資料

目前階段不要自行假設即時價格。
如果沒有資料，就明確告訴使用者資料尚未提供。
"""
)


result = Runner.run_sync(
    agent,
    "你好，請介紹你自己，以及你未來可以幫我做什麼？"
)


print(result.final_output)

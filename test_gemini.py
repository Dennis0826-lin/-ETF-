import os

from dotenv import load_dotenv
from google import genai


load_dotenv()


api_key = os.environ.get(
    "GEMINI_API_KEY"
)

if not api_key:

    raise ValueError(
        "找不到 GEMINI_API_KEY"
    )


print("=" * 60)
print("Gemini API Authentication Test")
print("=" * 60)

print(
    "API Key:",
    f"已設定 / 長度 {len(api_key)} / "
    f"開頭 {api_key[:4]}"
)

print()


try:

    client = genai.Client(
        api_key=api_key
    )

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents="請只回答 OK"
    )

    print("Gemini 回覆：")
    print(response.text)

    print()

    print("✅ Gemini API 測試成功")

except Exception as e:

    print("❌ Gemini API 測試失敗")

    print(
        type(e).__name__
    )

    print(
        str(e)
    )

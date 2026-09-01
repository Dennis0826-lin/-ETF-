import os
import requests
from dotenv import load_dotenv


# ==========================================
# 1. 載入 .env
# ==========================================

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

print("=" * 70)
print("Gemini REST API Authentication Test")
print("=" * 70)
print()

if not api_key:
    print("❌ GEMINI_API_KEY 沒有載入")
    raise SystemExit(1)

print(
    f"API Key 已載入："
    f"length={len(api_key)}, "
    f"prefix={api_key[:4]}..."
)

print()


# ==========================================
# 2. Gemini REST API
# ==========================================

url = (
    "https://generativelanguage.googleapis.com/"
    "v1beta/models/gemini-2.5-flash:generateContent"
)

headers = {
    "Content-Type": "application/json",
    "x-goog-api-key": api_key,
}

payload = {
    "contents": [
        {
            "parts": [
                {
                    "text": "請只回答 OK"
                }
            ]
        }
    ]
}


# ==========================================
# 3. 發送請求
# ==========================================

try:

    response = requests.post(
        url,
        headers=headers,
        json=payload,
        timeout=30,
    )

    print(
        f"HTTP Status Code: "
        f"{response.status_code}"
    )

    print()

    if response.status_code == 200:

        data = response.json()

        print("Gemini API 回應：")

        try:
            text = (
                data["candidates"][0]
                ["content"]["parts"][0]
                ["text"]
            )

            print(text)

        except Exception:

            print(data)

        print()
        print("✅ Gemini REST API 測試成功")

    else:

        print("❌ Gemini REST API 測試失敗")
        print()
        print(response.text)

except Exception as e:

    print("❌ HTTP 請求發生錯誤")
    print()
    print(type(e).__name__)
    print(str(e))


print()
print("=" * 70)
print("測試結束")
print("=" * 70)

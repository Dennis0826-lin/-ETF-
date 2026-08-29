import os
from langchain_community.document_loaders import (
    DirectoryLoader,
    PyPDFLoader,
    TextLoader,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS

def build_index():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    docs_dir = os.path.join(current_dir, "docs")
    
    if not os.path.exists(docs_dir):
        os.makedirs(docs_dir)
        print(f"📁 已建立 {docs_dir}/ 資料夾，請放入檔案。")
        test_file = os.path.join(docs_dir, "test.md")
        if not os.path.exists(test_file):
            with open(test_file, 'w', encoding='utf-8') as f:
                f.write("# 知識庫測試文件\n\n歡迎使用 RAG 向量查詢系統。")

    # 1. 讀取檔案
    documents = []
    print("🔍 正在載入文件...")
    md_loader = DirectoryLoader(docs_dir, glob="**/*.md", loader_cls=TextLoader, loader_kwargs={'encoding': 'utf-8'})
    documents.extend(md_loader.load())
    pdf_loader = DirectoryLoader(docs_dir, glob="**/*.pdf", loader_cls=PyPDFLoader)
    documents.extend(pdf_loader.load())

    if not documents:
        print("⚠️ docs/ 資料夾內沒有找到任何檔案。")
        return

    # 2. 切分文字
    print(f"✂️ 切分文字中 (共 {len(documents)} 個檔案)...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    splits = text_splitter.split_documents(documents)

    # 3. 轉向量（改用最穩定的通用模型名稱）
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        print("❌ 錯誤：找不到 GEMINI_API_KEY 環境變數！")
        return

    print("🔄 正在轉換向量...")
    # 使用 embedding-001 可相容所有 Gemini API 版本與權限
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=api_key
    )

    print("💾 建立 FAISS 向量庫...")
    vectorstore = FAISS.from_documents(splits, embeddings)
    
    index_path = os.path.join(current_dir, "faiss_index")
    vectorstore.save_local(index_path)
    print(f"✅ RAG 知識庫建立完成！已成功儲存至 '{index_path}' 資料夾。")

if __name__ == "__main__":
    build_index()

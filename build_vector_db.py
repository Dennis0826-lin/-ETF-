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
    # 強制手動設定 docs 目錄路徑，確保讀取正確
    current_dir = os.path.dirname(os.path.abspath(__file__))
    docs_dir = os.path.join(current_dir, "docs")
    
    if not os.path.exists(docs_dir):
        os.makedirs(docs_dir)
        print(f"📁 已建立 {docs_dir}/ 資料夾，請放入你的 PDF 或 Markdown 檔案後再執行此腳本。")
        # 建立一個測試檔案防止 DirectoryLoader 報錯
        test_file = os.path.join(docs_dir, "test.md")
        if not os.path.exists(test_file):
            with open(test_file, 'w', encoding='utf-8') as f:
                f.write("# 這是一個知識庫測試檔案\n\n您可以使用 search_personal_docs 工具來查詢我。")
            print(f"✅ 已建立測試檔案: {test_file}")

    # 1. 讀取 Markdown 與 PDF 檔案
    documents = []
    
    # 載入 .md 檔案
    print(f"🔍 正在載入 Markdown 檔案...")
    md_loader = DirectoryLoader(docs_dir, glob="**/*.md", loader_cls=TextLoader, loader_kwargs={'encoding': 'utf-8'})
    documents.extend(md_loader.load())
    
    # 載入 .pdf 檔案
    print(f"🔍 正在載入 PDF 檔案...")
    pdf_loader = DirectoryLoader(docs_dir, glob="**/*.pdf", loader_cls=PyPDFLoader)
    documents.extend(pdf_loader.load())

    if not documents:
        print("⚠️ docs/ 資料夾內沒有找到任何 .md 或 .pdf 檔案，將無法建立有效知識庫。")
        return

    # 2. 切分文章（Chunking）
    print(f"✂️ 正在進行文字切分 (共 {len(documents)} 個檔案)...")
    # 調整 Chunk size 和 Overlap 增加檢索效果
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    splits = text_splitter.split_documents(documents)
    print(f"   已切分為 {len(splits)} 個段落。")

    # 3. 轉向量並儲存至 FAISS
    # 強制指定模型名稱，去掉 'models/' 前綴，langchain-google-genai 應該正確處理它。
    # 錯誤原因是 'models/' 前綴導致 API URL 變成 models/models/text-embedding-004。
    model_name = "text-embedding-004"
    
    print(f"🔄 正在使用模型 '{model_name}' 將文字轉換為向量 (RAG Embeddings)...")
    
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    
    if not api_key:
        print("❌ 未偵測到 GEMINI_API_KEY 或 GOOGLE_API_KEY。請參考前一個步驟設定環境變數。")
        return

    embeddings = GoogleGenerativeAIEmbeddings(
        model=model_name, 
        google_api_key=api_key
    )

    print(f"💾 正在建立並儲存向量資料庫 (FAISS)...")
    vectorstore = FAISS.from_documents(splits, embeddings)
    
    # 手動指定儲存目錄路徑
    index_path = os.path.join(current_dir, "faiss_index")
    vectorstore.save_local(index_path)
    
    print(f"✅ RAG 知識庫建立完成！已儲存至 '{index_path}' 資料夾。")
    print(f"⚠️ **重要**：您需要將 '{index_path}' 資料夾中的檔案 (.faiss 和 .pkl) Commit & Push 到 GitHub，Agent 才能使用。")

if __name__ == "__main__":
    build_index()

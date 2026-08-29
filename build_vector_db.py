import os
import google.generativeai as genai
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.embeddings import Embeddings

# 自訂直接呼叫 Google 原生 API 的 Embedding Class，避免 LangChain 內部 404 Bug
class CustomGeminiEmbeddings(Embeddings):
    def __init__(self, api_key: str):
        genai.configure(api_key=api_key)
        self.model_name = "models/text-embedding-004"

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        # 批次處理 Avoid API Limit
        embeddings = []
        batch_size = 10
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            response = genai.embed_content(
                model=self.model_name,
                content=batch_texts,
                task_type="retrieval_document"
            )
            embeddings.extend(response['embedding'])
        return embeddings

    def embed_query(self, text: str) -> list[float]:
        response = genai.embed_content(
            model=self.model_name,
            content=text,
            task_type="retrieval_query"
        )
        return response['embedding']

def build_index():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    docs_dir = os.path.join(current_dir, "docs")
    
    if not os.path.exists(docs_dir):
        os.makedirs(docs_dir)
        print(f"📁 已建立 {docs_dir}/ 資料夾，請放入 Markdown 或 PDF 檔案。")
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

    # 3. 轉向量
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        print("❌ 錯誤：找不到 GEMINI_API_KEY 環境變數，請確認有成功 export 設定！")
        return

    print("🔄 正在使用 Google 原生 API 轉換向量...")
    embeddings = CustomGeminiEmbeddings(api_key=api_key)

    print("💾 建立 FAISS 向量庫...")
    vectorstore = FAISS.from_documents(splits, embeddings)
    
    index_path = os.path.join(current_dir, "faiss_index")
    vectorstore.save_local(index_path)
    print(f"✅ RAG 知識庫建立完成！已成功儲存至 '{index_path}' 資料夾。")

if __name__ == "__main__":
    build_index()

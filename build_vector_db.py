import os
import google.generativeai as genai
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.embeddings import Embeddings

# 自訂相容於 LangChain 的 Gemini Embeddings 類別
class CustomGeminiEmbeddings(Embeddings):
    def __init__(self, api_key: str, model_name: str = "models/text-embedding-004"):
        genai.configure(api_key=api_key)
        self.model_name = model_name

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        response = genai.embed_content(
            model=self.model_name,
            content=texts,
            task_type="retrieval_document"
        )
        return response['embedding']

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
        print(f"📁 已建立 {docs_dir}/ 資料夾，請放入檔案。")
        test_file = os.path.join(docs_dir, "test.md")
        if not os.path.exists(test_file):
            with open(test_file, 'w', encoding='utf-8') as f:
                f.write("# 個人知識庫測試文件\n\n歡迎使用 RAG 系統。")

    # 1. 讀取檔案
    documents = []
    print("🔍 正在載入文件...")
    md_loader = DirectoryLoader(docs_dir, glob="**/*.md", loader_cls=TextLoader, loader_kwargs={'encoding': 'utf-8'})
    documents.extend(md_loader.load())
    pdf_loader = DirectoryLoader(docs_dir, glob="**/*.pdf", loader_cls=PyPDFLoader)
    documents.extend(pdf_loader.load())

    if not documents:
        print("⚠️ docs/ 內無有效檔案。")
        return

    # 2. 切分文字
    print(f"✂️ 切分文字中 ({len(documents)} 個檔案)...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    splits = text_splitter.split_documents(documents)

    # 3. 轉向量
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        print("❌ 找不到 GEMINI_API_KEY 環境變數！")
        return

    print("🔄 使用 Google 原生 SDK 轉換向量中...")
    embeddings = CustomGeminiEmbeddings(api_key=api_key)

    print("💾 建立 FAISS 資料庫中...")
    vectorstore = FAISS.from_documents(splits, embeddings)
    
    index_path = os.path.join(current_dir, "faiss_index")
    vectorstore.save_local(index_path)
    print(f"✅ RAG 知識庫建立成功！已儲存至 '{index_path}'")

if __name__ == "__main__":
    build_index()

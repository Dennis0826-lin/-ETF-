cat << 'EOF' > build_vector_db.py
import os
from google import genai
from google.genai import types
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.embeddings import Embeddings

class DirectGeminiEmbeddings(Embeddings):
    def __init__(self, api_key: str):
        self.client = genai.Client(api_key=api_key)
        self.model = "text-embedding-004"

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        response = self.client.models.embed_content(
            model=self.model,
            contents=texts,
            config=types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT")
        )
        return [e.values for e in response.embeddings]

    def embed_query(self, text: str) -> list[float]:
        response = self.client.models.embed_content(
            model=self.model,
            contents=text,
            config=types.EmbedContentConfig(task_type="RETRIEVAL_QUERY")
        )
        return response.embeddings[0].values

def build_index():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    docs_dir = os.path.join(current_dir, "docs")
    
    if not os.path.exists(docs_dir):
        os.makedirs(docs_dir)
        print(f"📁 已建立 {docs_dir}/ 資料夾。")

    documents = []
    print("🔍 正在載入文件...")
    md_loader = DirectoryLoader(docs_dir, glob="**/*.md", loader_cls=TextLoader, loader_kwargs={'encoding': 'utf-8'})
    documents.extend(md_loader.load())
    pdf_loader = DirectoryLoader(docs_dir, glob="**/*.pdf", loader_cls=PyPDFLoader)
    documents.extend(pdf_loader.load())

    if not documents:
        print("⚠️ docs/ 資料夾內沒有找到任何檔案。")
        return

    print(f"✂️ 切分文字中 (共 {len(documents)} 個檔案)...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    splits = text_splitter.split_documents(documents)

    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        print("❌ 錯誤：找不到 GEMINI_API_KEY 或 GOOGLE_API_KEY 環境變數！")
        return

    print("🔄 正在轉換向量 (使用原生 Google GenAI SDK)...")
    embeddings = DirectGeminiEmbeddings(api_key=api_key)

    print("💾 建立 FAISS 向量庫...")
    vectorstore = FAISS.from_documents(splits, embeddings)
    
    index_path = os.path.join(current_dir, "faiss_index")
    vectorstore.save_local(index_path)
    print(f"✅ RAG 知識庫建立完成！已成功儲存至 '{index_path}' 資料夾。")

if __name__ == "__main__":
    build_index()
EOF

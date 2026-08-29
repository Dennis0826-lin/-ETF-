import os
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS

def build_index():
    docs_dir = "docs"
    if not os.path.exists(docs_dir):
        os.makedirs(docs_dir)
        print(f"📁 已建立 {docs_dir}/ 資料夾，請放入你的 PDF 或 Markdown 檔案後再執行此腳本。")
        return

    # 1. 讀取 Markdown 與 PDF 檔案
    documents = []
    
    # 載入 .md 檔案
    md_loader = DirectoryLoader(docs_dir, glob="**/*.md", loader_cls=TextLoader)
    documents.extend(md_loader.load())
    
    # 載入 .pdf 檔案
    pdf_loader = DirectoryLoader(docs_dir, glob="**/*.pdf", loader_cls=PyPDFLoader)
    documents.extend(pdf_loader.load())

    if not documents:
        print("⚠️ docs/ 資料夾內沒有找到任何 .md 或 .pdf 檔案！")
        return

    # 2. 切分文章（Chunking）
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    splits = text_splitter.split_documents(documents)

    # 3. 轉向量並儲存至 FAISS
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/text-embedding-004", 
        google_api_key=api_key
    )

    vectorstore = FAISS.from_documents(splits, embeddings)
    vectorstore.save_local("faiss_index")
    print("✅ 向量資料庫建立完成！已儲存至 faiss_index/ 資料夾。")

if __name__ == "__main__":
    build_index()

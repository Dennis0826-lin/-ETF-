import os
from langchain_community.document_loaders import (
    DirectoryLoader,
    PyPDFLoader,
    TextLoader,
    UnstructuredWordDocumentLoader,
)
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


def build_index():
    docs_dir = "docs"
    if not os.path.exists(docs_dir):
        os.makedirs(docs_dir)
        print(
            f"📁 已建立 {docs_dir}/ 資料夾，請放入你的檔案後再執行此腳本。"
        )
        return

    documents = []

    # 1. 載入 .md 與 .txt 檔案
    md_loader = DirectoryLoader(
        docs_dir, glob="**/*.md", loader_cls=TextLoader
    )
    txt_loader = DirectoryLoader(
        docs_dir, glob="**/*.txt", loader_cls=TextLoader
    )
    documents.extend(md_loader.load())
    documents.extend(txt_loader.load())

    # 2. 載入 .pdf 檔案
    pdf_loader = DirectoryLoader(
        docs_dir, glob="**/*.pdf", loader_cls=PyPDFLoader
    )
    documents.extend(pdf_loader.load())

    # 3. 載入 .docx 檔案
    docx_loader = DirectoryLoader(
        docs_dir, glob="**/*.docx", loader_cls=UnstructuredWordDocumentLoader
    )
    try:
        documents.extend(docx_loader.load())
    except Exception as e:
        print(f"⚠️ 載入 Word 檔案時略過部分未支援格式: {str(e)}")

    if not documents:
        print("⚠️ docs/ 資料夾內沒有找到任何支援的檔案！")
        return

    # 4. 文字切分 (Chunking)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500, chunk_overlap=50
    )
    splits = text_splitter.split_documents(documents)

    # 5. 轉向量並儲存至 FAISS
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get(
        "GOOGLE_API_KEY"
    )
   # 找到這一段，修改為：
    embeddings = GoogleGenerativeAIEmbeddings(
        model="text-embedding-004", 
        google_api_key=api_key
    )

    vectorstore = FAISS.from_documents(splits, embeddings)
    vectorstore.save_local("faiss_index")
    print("✅ 向量資料庫建立完成！已儲存至 faiss_index/ 資料夾。")


if __name__ == "__main__":
    build_index()

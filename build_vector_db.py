cat << 'EOF' > build_vector_db.py
import os
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

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

    print("🔄 正在使用本地 HuggingFace 模型轉換向量 (不需 API Key)...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    print("💾 建立 FAISS 向量庫...")
    vectorstore = FAISS.from_documents(splits, embeddings)
    
    index_path = os.path.join(current_dir, "faiss_index")
    vectorstore.save_local(index_path)
    print(f"✅ RAG 知識庫建立完成！已成功儲存至 '{index_path}' 資料夾。")

if __name__ == "__main__":
    build_index()
EOF

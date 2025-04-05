# kb_builder.py
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import SentenceTransformerEmbeddings

def build_vector_store(filepath="data/agri_guide.txt"):
    loader = TextLoader(filepath)
    documents = loader.load()

    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    db = FAISS.from_documents(documents, embeddings)

    db.save_local("vector_store")

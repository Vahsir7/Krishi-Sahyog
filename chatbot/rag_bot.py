# chatbot/rag_bot.py
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain.chains.question_answering import load_qa_chain
from langchain_community.llms import OpenAI  # Or use Ollama for local LLMs

def load_vector_store():
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    return FAISS.load_local("vector_store", embeddings)

def get_answer(query):
    db = load_vector_store()
    docs = db.similarity_search(query)

    llm = OpenAI()  # Replace with Ollama if local
    chain = load_qa_chain(llm, chain_type="stuff")

    return chain.run(input_documents=docs, question=query)

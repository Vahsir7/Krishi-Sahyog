import subprocess
import pandas as pd
import numpy as np
import faiss
from sklearn.preprocessing import StandardScaler
from sentence_transformers import SentenceTransformer

# Load datasets
def load_farmer_data():
    return pd.read_csv("data/farmer_advisor_dataset.csv")

def load_market_data():
    return pd.read_csv("data/market_researcher_dataset.csv")

# Sentence transformer model
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# Normalize and embed context using FAISS
def prepare_faiss_index(df):
    combined = df.astype(str).agg(" ".join, axis=1).values
    embeddings = embedder.encode(combined)
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings))
    return index, combined

def retrieve_similar_context(query, index, combined_texts, k=3):
    query_vec = embedder.encode([query])
    distances, indices = index.search(np.array(query_vec), k)
    return [combined_texts[i] for i in indices[0]]

OLLAMA_PATH = "/usr/local/bin/ollama"

def ask_ollama_crop_advisor(query):
    df = load_farmer_data()
    index, combined = prepare_faiss_index(df)
    top_matches = retrieve_similar_context(query, index, combined)    
    context = "\n".join(top_matches)
    prompt = f"""
You are an expert crop advisor trained on agricultural data.

Relevant Context:
{context}

A farmer asks: "{query}"

Based on the above data and conditions, suggest 1-3 suitable crops along with reasons for each suggestion.
"""
    try:
        result = subprocess.run(
            [OLLAMA_PATH, "run", "llama2:7b-chat"],
            input=prompt,
            capture_output=True,
            text=True
        )
        if result.returncode != 0 or not result.stdout.strip():
            return "Error occurred while getting advice."
        return result.stdout.strip()
    except Exception as e:
        print("Error:", e)
        return "Error occurred while getting advice."

# Ask market advisor
def ask_ollama_market_advisor(query):
    df = load_market_data()
    index, combined = prepare_faiss_index(df)
    top_matches = retrieve_similar_context(query, index, combined)

    context = "\n".join(top_matches)
    prompt = f"""
You are an expert market advisor trained on agricultural market data.

Relevant Context:
{context}

A user asks: "{query}"

Based on the dataset, provide market insights and recommendations regarding crop pricing, demand trends, and competitor analysis.
"""
    try:
        result = subprocess.run(
            [OLLAMA_PATH, "run", "llama2:7b-chat"],
            input=prompt,
            capture_output=True,
            text=True
        )
        if result.returncode != 0 or not result.stdout.strip():
            return "Error occurred while getting advice."
        return result.stdout.strip()
    except Exception as e:
        print("Error:", e)
        return "Error occurred while getting advice."

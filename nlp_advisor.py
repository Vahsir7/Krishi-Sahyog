import subprocess
import pandas as pd
import numpy as np
import faiss
from sklearn.preprocessing import StandardScaler
from sentence_transformers import SentenceTransformer
import os
import requests
import json
from dotenv import load_dotenv

load_dotenv()

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

OLLAMA_URL = os.environ.get("OLLAMA_URL")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL")

def _query_ollama_api(prompt):
    """Helper function to query the Ollama API."""
    if not OLLAMA_URL or not OLLAMA_MODEL:
        error_msg = "[Error: OLLAMA_URL or OLLAMA_MODEL environment variable not set.]"
        print(error_msg)
        return error_msg

    api_url = f"{OLLAMA_URL}/api/generate"
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False
    }
    try:
        response = requests.post(api_url, json=payload, timeout=60)
        response.raise_for_status()
        data = response.json()
        ollama_response = data.get("response", "").strip()
        if not ollama_response:
            return "Error occurred while getting advice: Empty response from model."
        return ollama_response
    except requests.exceptions.RequestException as e:
        print("Error:", e)
        return f"Error occurred while getting advice: {e}"
    except Exception as e:
        print("Error:", e)
        return "An unexpected error occurred while getting advice."

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
    return _query_ollama_api(prompt)

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
    return _query_ollama_api(prompt)

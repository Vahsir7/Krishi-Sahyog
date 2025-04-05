import pandas as pd
import subprocess

# Set the path to the Ollama executable (adjust if needed)
OLLAMA_PATH = "/usr/local/bin/ollama"

# -------------------------------
# Load CSV Data Once at Startup
# -------------------------------
def load_farmer_data():
    return pd.read_csv("data/farmer_advisor_dataset.csv")

def load_market_data():
    return pd.read_csv("data/market_researcher_dataset.csv")

# Global variables for KBs
FARMER_KB = []
MARKET_KB = []

def build_farmer_kb():
    df = load_farmer_data()
    kb = []
    for _, row in df.iterrows():
        entry = (
            f"Crop: {row['Crop_Type']}. "
            f"Soil pH: {row['Soil_pH']}, Moisture: {row['Soil_Moisture']}%, "
            f"Temperature: {row['Temperature_C']}°C, Rainfall: {row['Rainfall_mm']} mm, "
            f"Yield: {row['Crop_Yield_ton']} tons."
        )
        kb.append(entry)
    print("Farmer KB built successfully.")
    return kb

def build_market_kb():
    df = load_market_data()
    kb = []
    for _, row in df.iterrows():
        entry = (
            f"Crop: {row['Product']}. "
            f"Market Price: {row['Market_Price_per_ton']}, Demand Index: {row['Demand_Index']}, "
            f"Supply Index: {row['Supply_Index']}, Competitor Price: {row['Competitor_Price_per_ton']}, "
            f"Weather Impact Score: {row['Weather_Impact_Score']}, Seasonal Factor: {row['Seasonal_Factor']}."
        )
        kb.append(entry)
    print("Market KB built successfully.")
    return kb

# Build KBs at startup
FARMER_KB = build_farmer_kb()
MARKET_KB = build_market_kb()
print("Knowledge Bases Loaded:")
print(f"Farmer KB: {len(FARMER_KB)} entries")
print(f"Market KB: {len(MARKET_KB)} entries")

# -------------------------------
# Simple Retrieval from the KB by Keyword Matching
# -------------------------------
def retrieve_context(query, kb, top_n=3):
    query_words = set(query.lower().split())
    scored = []
    for entry in kb:
        entry_words = set(entry.lower().split())
        score = len(query_words.intersection(entry_words))
        scored.append((score, entry))
    scored.sort(key=lambda x: x[0], reverse=True)
    context_entries = [entry for score, entry in scored if score > 0][:top_n]
    return "\n".join(context_entries)

# -------------------------------
# Query Ollama with a Constructed Prompt
# -------------------------------
def query_ollama(prompt):
    print(f"Querying Ollama with prompt: {prompt}")
    result = subprocess.run(
        [OLLAMA_PATH, "run", "llama2:7b-chat"],
        input=prompt,
        capture_output=True,
        text=True
    )
    if result.returncode != 0 or not result.stdout.strip():
        return "Error: Model did not return advice."
    print(f"Ollama response: {result.stdout.strip()}")
    return result.stdout.strip()

# -------------------------------
# Main Functions for Each Agent
# -------------------------------
def get_farmer_advice(query):
    context = retrieve_context(query, FARMER_KB)
    prompt = f"""
You are an expert crop advisor.
Historic Farmer Data Context:
{context}
User Query: "{query}"
Based on the historical data, provide a detailed crop recommendation with reasoning.
"""
    return query_ollama(prompt)

def get_market_advice(query):
    context = retrieve_context(query, MARKET_KB)
    prompt = f"""
You are an expert market advisor.
Historic Market Data Context:
{context}
User Query: "{query}"
Based on the historical market data, provide detailed market insights and recommendations.
"""
    return query_ollama(prompt)

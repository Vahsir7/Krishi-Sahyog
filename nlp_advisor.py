# nlp_advisor.py
import subprocess
import pandas as pd

def load_farmer_data():
    # Adjust the path if needed
    return pd.read_csv("data/farmer_advisor_dataset.csv")

def load_market_data():
    return pd.read_csv("data/market_researcher_dataset.csv")

def generate_context_for_crop():
    df = load_farmer_data()
    # For demonstration, take a small sample or key statistics.
    # You might want to generate a more refined summary in a production system.
    sample = df.head(5).to_string(index=False)
    context = f"Farmer Dataset Sample:\n{sample}"
    return context

def generate_context_for_market():
    df = load_market_data()
    sample = df.head(5).to_string(index=False)
    context = f"Market Dataset Sample:\n{sample}"
    return context

def ask_ollama_crop_advisor(query):
    context = generate_context_for_crop()
    prompt = f"""
You are an expert crop advisor trained on the following dataset:
{context}

A farmer asks: "{query}"
Based on the dataset and the conditions provided, suggest 1-3 suitable crops along with reasons for each suggestion.
"""
    result = subprocess.run(
        ["ollama", "run", "llama3", "--prompt", prompt],
        capture_output=True,
        text=True
    )
    return result.stdout.strip()

def ask_ollama_market_advisor(query):
    context = generate_context_for_market()
    prompt = f"""
You are an expert market advisor trained on the following dataset:
{context}

A user asks: "{query}"
Based on the dataset, provide market insights and recommendations regarding crop pricing, demand trends, and competitor analysis.
"""
    result = subprocess.run(
        ["ollama", "run", "llama3", "--prompt", prompt],
        capture_output=True,
        text=True
    )
    return result.stdout.strip()

# nlp_advisor.py
import subprocess
import pandas as pd

# Functions to load your datasets
def load_farmer_data():
    return pd.read_csv("data/farmer_advisor_dataset.csv")

def load_market_data():
    return pd.read_csv("data/market_researcher_dataset.csv")

# Generate a brief context from the farmer dataset (customize as needed)
def generate_context_for_crop():
    df = load_farmer_data()
    sample = df.head(5).to_string(index=False)
    context = f"Farmer Dataset Sample:\n{sample}"
    return context

# Generate a brief context from the market dataset (customize as needed)
def generate_context_for_market():
    df = load_market_data()
    sample = df.head(5).to_string(index=False)
    context = f"Market Dataset Sample:\n{sample}"
    return context

# Path to the Ollama executable; adjust if necessary
OLLAMA_PATH = "/usr/local/bin/ollama"  # Ensure 'ollama' is in your PATH; otherwise, provide the full path

# Function to query the crop advisor using the llama2-7b-chat model
def ask_ollama_crop_advisor(query):
    context = generate_context_for_crop()
    prompt = f"""
You are an expert crop advisor trained on agricultural data.

Dataset Context:
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
        # Check if the subprocess ran successfully
        if result.returncode != 0:
            print("Subprocess error:", result.stderr.strip())
            return "Error occurred while getting advice."
        # Check if the output is empty
        if not result.stdout.strip():
            print("Subprocess output is empty.")
            return "No advice returned from model."
        # Print the output for debugging
        print("Subprocess output:", result.stdout.strip())  # Debug print
        output = result.stdout.strip()
        if not output:
            # Print stderr to help diagnose issues
            print("Subprocess stderr:", result.stderr.strip())
        print("Ollama output:", output)  # Debug print
        return output if output else "No advice returned from model."
    except Exception as e:
        print("Error:", e)
        return "Error occurred while getting advice."

# Function to query the market advisor using the llama2-7b-chat model
def ask_ollama_market_advisor(query):
    context = generate_context_for_market()
    prompt = f"""
You are an expert market advisor trained on agricultural market data.

Dataset Context:
{context}

A user asks: "{query}"

Based on the dataset, provide market insights and recommendations regarding crop pricing, demand trends, and competitor analysis.
"""
    result = subprocess.run(
        [OLLAMA_PATH, "run", "llama2:7b-chat"],
        input=prompt,
        capture_output=True,
        text=True
    )

    return result.stdout.strip()

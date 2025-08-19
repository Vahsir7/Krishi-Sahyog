import os
import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from langdetect import detect, LangDetectException
from deep_translator import GoogleTranslator

# --- Configuration ---
# This will be dynamically set from app.py to handle different datasets
MODEL_NAME = 'google/muril-base-cased'

# --- Global In-Memory Storage ---
# We use a dictionary to hold models and data for different advisors (farmer, market)
nlp_resources = {}

def initialize_advisor(advisor_name: str, data_file: str):
    """
    Loads the model, data, and FAISS index for a specific advisor.
    """
    global nlp_resources
    
    if advisor_name in nlp_resources:
        print(f"'{advisor_name}' advisor already initialized.")
        return

    try:
        print(f"--- Initializing '{advisor_name}' Advisor ---")
        
        # Load the model only once
        if 'model' not in nlp_resources:
            print(f"Loading sentence transformer model: {MODEL_NAME}...")
            nlp_resources['model'] = SentenceTransformer(MODEL_NAME)
        
        faiss_index_file = f"{advisor_name}.index"
        
        # Create FAISS index if it doesn't exist
        if not os.path.exists(faiss_index_file):
            print(f"FAISS index for '{advisor_name}' not found. Creating one...")
            df_temp = pd.read_csv(data_file)
            # Use a generic 'question' or 'query' column if available, otherwise combine all
            if 'question' in df_temp.columns:
                text_to_embed = df_temp['question'].dropna().tolist()
            elif 'query' in df_temp.columns:
                 text_to_embed = df_temp['query'].dropna().tolist()
            else: # Fallback: combine all columns to create context strings
                df_temp.fillna('', inplace=True)
                text_to_embed = df_temp.astype(str).agg(' '.join, axis=1).tolist()

            embeddings = nlp_resources['model'].encode(text_to_embed)
            dim = embeddings.shape[1]
            index_temp = faiss.IndexFlatL2(dim)
            index_temp.add(np.array(embeddings))
            faiss.write_index(index_temp, faiss_index_file)

        # Load resources into memory
        resources = {
            "index": faiss.read_index(faiss_index_file),
            "dataframe": pd.read_csv(data_file)
        }
        nlp_resources[advisor_name] = resources
        print(f"--- '{advisor_name}' Advisor Ready ---")

    except Exception as e:
        print(f"Fatal Error initializing '{advisor_name}' advisor: {e}")

def get_base_answer(advisor_name: str, user_message: str) -> (str, str):
    """
    Uses MURIL and FAISS to get the ground-truth answer and detects language.
    """
    if advisor_name not in nlp_resources or 'model' not in nlp_resources:
        return "Error: NLP system not initialized for this advisor.", "en"

    resources = nlp_resources[advisor_name]
    model = nlp_resources['model']
    faiss_index = resources['index']
    df = resources['dataframe']

    try:
        target_lang = detect(user_message)
    except LangDetectException:
        target_lang = 'en'

    query_embedding = model.encode([user_message])
    _, indices = faiss_index.search(np.array(query_embedding), 1)
    best_match_index = indices[0][0]
    
    # Heuristically find the best 'answer' or 'response' column
    if 'answer' in df.columns:
        retrieved_english_answer = df.iloc[best_match_index]['answer']
    elif 'response' in df.columns:
        retrieved_english_answer = df.iloc[best_match_index]['response']
    else: # Fallback to just returning the whole row as context
        retrieved_english_answer = ' '.join(df.iloc[best_match_index].astype(str).tolist())
    
    if target_lang != 'en':
        try:
            translated_answer = GoogleTranslator(source='auto', target=target_lang).translate(retrieved_english_answer)
            return translated_answer or retrieved_english_answer, target_lang
        except Exception:
            return retrieved_english_answer, target_lang
    
    return retrieved_english_answer, target_lang

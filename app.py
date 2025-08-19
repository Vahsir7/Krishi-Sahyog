import os
import faiss
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template
from sentence_transformers import SentenceTransformer
from flask_cors import CORS
# --- Imports for Language Detection and Translation (Reverted to deep-translator) ---
from langdetect import detect, LangDetectException
from deep_translator import GoogleTranslator

# --- Initialization ---
app = Flask(__name__)
CORS(app)

# --- Configuration ---
DATA_FILE = 'farming_qna_english.csv' # Using a new file name to avoid confusion
FAISS_INDEX_FILE = 'farming_qna.index'
MODEL_NAME = 'google/muril-base-cased'

# --- Load Sentence Transformer Model ---
try:
    print(f"Loading sentence transformer model: {MODEL_NAME}...")
    model = SentenceTransformer(MODEL_NAME)
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

# --- Prepare Data and FAISS Index ---
# This section now works with a simple English-only CSV
if os.path.exists(DATA_FILE) and not os.path.exists(FAISS_INDEX_FILE):
    try:
        print("Data file found, but FAISS index not found. Creating index...")
        df = pd.read_csv(DATA_FILE)
        # Ensure the 'question' column exists
        if 'question' not in df.columns:
            raise ValueError("CSV must have a 'question' column containing English questions.")
        
        questions = df['question'].tolist()
        print(f"Generating embeddings for {len(questions)} English questions...")
        question_embeddings = model.encode(questions, convert_to_tensor=True, show_progress_bar=True)
        question_embeddings = question_embeddings.cpu().numpy()
        
        embedding_dim = question_embeddings.shape[1]
        index = faiss.IndexFlatL2(embedding_dim)
        index.add(question_embeddings)
        
        faiss.write_index(index, FAISS_INDEX_FILE)
        print(f"FAISS index created and saved to {FAISS_INDEX_FILE}.")
    except Exception as e:
        print(f"Error creating FAISS index: {e}")
        exit()

# --- Load FAISS Index and Data ---
try:
    print("Loading FAISS index and data file...")
    index = faiss.read_index(FAISS_INDEX_FILE)
    df = pd.read_csv(DATA_FILE)
    print("FAISS index and data loaded successfully.")
except Exception as e:
    print(f"Could not load FAISS index or data file: {e}")
    index = None
    df = None

# --- Flask Routes ---
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    if index is None or df is None:
        return jsonify({"error": "Server is not ready. Index or data not loaded."}), 500

    try:
        user_message = request.json.get('message')
        if not user_message:
            return jsonify({"error": "Message cannot be empty."}), 400

        print(f"Received message: {user_message}")

        # --- STEP 1: DETECT USER'S LANGUAGE ---
        try:
            target_lang = detect(user_message)
            print(f"Detected language: {target_lang}")
        except LangDetectException:
            print("Could not detect language, defaulting to English ('en').")
            target_lang = 'en'

        # --- STEP 2: RETRIEVE BEST ENGLISH ANSWER ---
        query_embedding = model.encode([user_message])
        k = 1
        distances, indices = index.search(query_embedding, k)
        best_match_index = indices[0][0]
        
        # Retrieve the English answer from the dataframe
        retrieved_english_answer = df.iloc[best_match_index]['answer']
        
        # --- STEP 3: TRANSLATE THE ANSWER BACK TO USER'S LANGUAGE (Using deep-translator) ---
        final_response = retrieved_english_answer
        # Only translate if the target language is not English
        if target_lang != 'en':
            try:
                print(f"Translating answer to '{target_lang}'...")
                # Using the 'deep-translator' library now for better quality
                translated_answer = GoogleTranslator(source='auto', target=target_lang).translate(retrieved_english_answer)
                
                if translated_answer:
                    final_response = translated_answer
                else:
                    # Fallback in case translation returns an empty string
                    final_response = retrieved_english_answer
            except Exception as trans_error:
                print(f"Translation error: {trans_error}")
                # If translation fails, just send the original English answer
                final_response = retrieved_english_answer

        print(f"Sending response: {final_response}")
        return jsonify({"response": final_response})

    except Exception as e:
        print(f"An error occurred during chat processing: {e}")
        return jsonify({"error": "An internal error occurred."}), 500

# --- Main Execution ---
if __name__ == '__main__':
    if not os.path.exists(DATA_FILE):
        print(f"'{DATA_FILE}' not found. Creating a sample file.")
        # Simple English-only structure
        sample_data = {
            'question': [
                "What is the best season to grow wheat?",
                "How to control pests in rice crop?",
                "What is drip irrigation?",
                "Best fertilizer for tomatoes?",
                "Which soil is suitable for wheat?",
                "What is the best time to sow wheat?"
            ],
            'answer': [
                "The best season to grow wheat is the winter (Rabi season), typically from October to December.",
                "You can use neem oil spray, introduce natural predators like ladybugs, or use appropriate bio-pesticides.",
                "Drip irrigation is a micro-irrigation system that saves water and nutrients by allowing water to drip slowly to the roots of plants.",
                "A balanced fertilizer with a higher phosphorus and potassium content (like a 5-10-10 formula) is great for tomatoes.",
                "Alluvial soil is considered ideal for wheat cultivation.",
                "The best time for sowing wheat is during moderate temperature and after moderate rainfall."
            ]
        }
        pd.DataFrame(sample_data).to_csv(DATA_FILE, index=False)
        print(f"Sample data file '{DATA_FILE}' created. Please delete the old 'farming_qna.index' file if it exists and restart the server.")
        exit()

    app.run(host='0.0.0.0', port=5000, debug=True)

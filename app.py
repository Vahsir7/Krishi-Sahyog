import os
import faiss
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template
from sentence_transformers import SentenceTransformer
from flask_cors import CORS
# --- New Imports for Translation ---
from langdetect import detect
from deep_translator import GoogleTranslator

# --- Initialization ---
app = Flask(__name__)
CORS(app)

# --- Configuration ---
DATA_FILE = 'farming_qna.csv'
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
# (This section remains unchanged)
if os.path.exists(DATA_FILE) and not os.path.exists(FAISS_INDEX_FILE):
    try:
        print("Data file found, but FAISS index not found. Creating index...")
        df = pd.read_csv(DATA_FILE)
        if 'question' not in df.columns:
            raise ValueError("CSV must have a 'question' column.")
        questions = df['question'].tolist()
        print(f"Generating embeddings for {len(questions)} questions...")
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
# (This section remains unchanged)
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

        # --- STEP 1: RETRIEVAL (Unchanged) ---
        # Find the most relevant answer from our database
        query_embedding = model.encode([user_message])
        k = 1
        distances, indices = index.search(query_embedding, k)
        best_match_index = indices[0][0]
        retrieved_answer = df.iloc[best_match_index]['answer']
        
        # --- STEP 2: LANGUAGE DETECTION & TRANSLATION (New Logic) ---
        final_response = retrieved_answer
        try:
            # Detect the language of the user's query
            detected_lang = detect(user_message)
            print(f"Detected language: {detected_lang}")

            # Translate the retrieved answer to the detected language
            # The translator will automatically detect the source language of our answer
            translated_answer = GoogleTranslator(source='auto', target=detected_lang).translate(retrieved_answer)
            
            if translated_answer:
                final_response = translated_answer
            else:
                # Fallback in case translation returns empty
                final_response = retrieved_answer

        except Exception as lang_error:
            # If language detection or translation fails, just return the original answer
            print(f"Language detection/translation error: {lang_error}")
            final_response = retrieved_answer


        print(f"Sending response: {final_response}")
        return jsonify({"response": final_response})

    except Exception as e:
        print(f"An error occurred during chat processing: {e}")
        return jsonify({"error": "An internal error occurred."}), 500

# --- Main Execution ---
# (This section remains unchanged)
if __name__ == '__main__':
    if not os.path.exists(DATA_FILE):
        print(f"'{DATA_FILE}' not found. Creating a sample file.")
        sample_data = {
            'question': [
                "What is the best season to grow wheat?",
                "गेहूँ उगाने का सबसे अच्छा मौसम कौन सा है?",
                "How to control pests in rice crop?",
                "चावल की फसल में कीटों को कैसे नियंत्रित करें?",
                "What is drip irrigation?",
                "ड्रिप सिंचाई क्या है?",
                "Best fertilizer for tomatoes?",
                "टमाटर के लिए सबसे अच्छी खाद कौन सी है?"
            ],
            'answer': [
                "The best season to grow wheat is the winter (Rabi season), typically from October to December.",
                "गेहूँ उगाने का सबसे अच्छा मौसम सर्दियों (रबी मौसम) का होता है, आमतौर पर अक्टूबर से दिसंबर तक।",
                "You can use neem oil spray, introduce natural predators like ladybugs, or use appropriate bio-pesticides.",
                "आप नीम के तेल का स्प्रे इस्तेमाल कर सकते हैं, लेडीबग जैसे प्राकृतिक शिकारियों को छोड़ सकते हैं, या उचित जैव-कीटनाशकों का उपयोग कर सकते हैं।",
                "Drip irrigation is a micro-irrigation system that saves water and nutrients by allowing water to drip slowly to the roots of plants.",
                "ड्रिप सिंचाई एक सूक्ष्म सिंचाई प्रणाली है जो पानी और पोषक तत्वों की बचत करती है, जिससे पानी धीरे-धीरे पौधों की जड़ों तक टपकता है।",
                "A balanced fertilizer with a higher phosphorus and potassium content (like a 5-10-10 formula) is great for tomatoes.",
                "टमाटर के लिए उच्च फास्फोरस और पोटेशियम सामग्री (जैसे 5-10-10 फॉर्मूला) वाला संतुलित उर्वरक बहुत अच्छा होता है।"
            ]
        }
        pd.DataFrame(sample_data).to_csv(DATA_FILE, index=False)
        print("Sample data file created. Please restart the server.")
        exit()

    app.run(host='0.0.0.0', port=5000, debug=True)

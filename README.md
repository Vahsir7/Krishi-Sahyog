---
title: Krishi Sahyog
emoji: 🌾
colorFrom: green
colorTo: yellow
sdk: docker
main: app.py
---
# Krishi Sahyog 🌱🤖

**Krishi Sahyog** is a multi-agent AI advisor for smart farming.
It empowers farmers with **data-driven insights, personalized crop advice, and real-time market trends**.
The system integrates **retrieval-augmented generation (RAG)** with multilingual NLP so farmers can interact in their **local language**, making technology more accessible.

---

## 🚀 New Features

* **Farmer Advice Assistant** – Region & crop-specific recommendations based on datasets.
* **Market Insight Module** – Provides farmers with market trends, pricing, and demand insights.
* **Multilingual RAG Bot** – Supports Indian languages using:

  * [Ollama](https://ollama.ai/) for efficient local model serving
  * [Google MuRIL](https://huggingface.co/google/muril-base-cased) for multilingual contextual understanding
  * [Sentence Transformers](https://www.sbert.net/) for semantic embeddings
  * [FAISS](https://github.com/facebookresearch/faiss) for fast similarity search

---

## 📌 Features

* 🤖 **Multi-Agent Architecture:** Specialized agents for farming insights, NLP, and market analysis.
* 📊 **Data-Driven Insights:** SQLite-backed agricultural datasets for tailored recommendations.
* 🗣️ **Multilingual NLP:** Farmers can ask questions in English or Indic languages.
* 📈 **Market Trends & Analysis:** Helps farmers make informed selling decisions.
* ⚡ **Integration with Ollama Agents:** For advanced, modular AI reasoning.
* 💻 **User-Friendly Interface:** Built with HTML + Python backend for easy accessibility.

---

## 🗂️ Project Structure

```plaintext
Krishi-Sahyog/
├── agents/                # AI agents for different tasks
├── data/                  # Agricultural datasets
├── static/                # CSS, JavaScript, images
├── templates/             # HTML templates
├── app.py                 # Main application server
├── nlp_advisor.py         # NLP advisor agent
├── requirements.txt       # Python dependencies
└── smart_agri.db          # SQLite database
```

---

## ⚙️ Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/Vahsir7/Krishi-Sahyog.git
   cd Krishi-Sahyog
   ```

2. **Create a virtual environment (recommended):**

   ```bash
   python -m venv venv
   source venv/bin/activate
   # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

4. **Install Ollama (for model serving):**

   * Follow instructions at [https://ollama.ai/download](https://ollama.ai/download)
   * Ensure models are available locally (e.g., `llama2`, `mistral`, or custom models).

5. **Database setup:**
   The repo includes a prebuilt `smart_agri.db`.
   To customize or reinitialize, use scripts in the `agents/` folder.

6. **Install FAISS (for vector search):**

   ```bash
   pip install faiss-cpu
   ```

   (or `faiss-gpu` if you have CUDA support).

7. **Install Sentence Transformers & MuRIL:**

   ```bash
   pip install sentence-transformers
   pip install torch torchvision torchaudio
   ```

   MuRIL can be loaded from TensorFlow Hub in your code.

---

## ▶️ Usage

1. **Run the server:**

   ```bash
   uvicorn app:app --host 0.0.0.0 --port 8001 --reload
   ```

2. **Open in browser:**
   [http://127.0.0.1:8001](http://127.0.0.1:8001)

   Or use your system’s IPv4 address + port `8001` on any device in the same network.

3. **Ask questions:**

   * Crop-specific queries (e.g., “Best fertilizer for paddy in West Bengal?”)
   * Market queries (e.g., “What is the current trend for wheat prices?”)
   * General farming advice in **English or local language**

---

## 🤝 Contributing

Contributions are welcome!
Please fork the repo, open an issue, or submit a pull request.

---

## 📬 Contact

For questions, suggestions, or collaborations:
**Email:** [rishavbairagya@gmail.com](mailto:rishavbairagya@gmail.com)
**Phone:** +91 9477455599

---

🌾 *Happy Farming with AI!* 🌾


# Krishi Sahyog

**Krishi Sahyog** is a multi-agent AI advisor for smart farming. This project leverages your agricultural datasets and a SQLite database to help farmers make data-driven decisions. The system uses multiple cooperating agents—each with specialized tasks—to analyze data and provide actionable insights for improving farm productivity and sustainability.

---

## Table of Contents

- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

---

## Features

- **Multi-Agent Architecture:** Multiple specialized agents work together to analyze data and provide advice.
- **Data-Driven Insights:** Integrates your datasets with SQLite to deliver tailored farming recommendations.
- **Natural Language Processing:** The NLP advisor agent interprets user queries and provides context-specific responses.
- **Integration of Ollama Agents:** Leverages the capabilities of Ollama agents for enhanced data processing and analysis.
- **User-Friendly Interface:** Built with HTML and Python to ensure an accessible user experience.
- **Modular Design:** Easily extend or modify agents and functionalities to suit your specific agricultural needs.


---

## Project Structure

```plaintext
Smart-Agriculture-AI/
├── agents/                # Contains individual agent scripts for various tasks
├── data/                  # Place your datasets here
├── static/                # Static assets (CSS, JavaScript, images)
├── templates/             # HTML templates for the web interface
├── app.py                 # Main application file to run the server
├── nlp_advisor.py         # NLP-based advisor agent for processing natural language queries
├── requirements.txt       # Python dependencies required for the project
└── smart_agri.db          # SQLite database file for storing and querying data
```

---

## Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/Vahsir7/Smart-Agriculture-Ai.git
   cd Smart-Agriculture-Ai
   ```

2. **Create a virtual environment (optional but recommended):**

   ```bash
   python -m venv venv
   source venv/bin/activate   
   # On Windows use: venv\Scripts\activate
   ```

3. **Install the dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

4. **Set up your SQLite database:**

   The repository includes a pre-built `smart_agri.db`. If you need to update or initialize the database schema, refer to the project documentation or scripts within the `agents` folder.

---

## Usage

1. **Run the application:**

   ```bash
   uvicorn app:app --host 0.0.0.0 --port 8000 --reload
   ```

2. **Access the application:**

   Open your web browser and navigate to [http://127.0.0.1:8000](http://127.0.0.1:8000) to start using the AI advisor for smart farming.

3. **Interacting with Agents:**

   The application coordinates multiple agents that analyze the data from your datasets. For instance, the `nlp_advisor.py` processes natural language queries to provide context-specific recommendations.

---

## Contact

For any questions, suggestions, or issues, please contact:
**Email:** [rishavbairagya@gmail.com](mailto:rishavbairagya@gmail.com)  
**Phone:** +91 9477455599

---

Happy Farming with AI!!

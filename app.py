import sys
import os
import re
import asyncio
import sqlite3 
import datetime 
import json 
from fastapi import FastAPI, Request, Form, HTTPException 
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates


project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)
try:
    import agents.custom_agent as agent 
    if not hasattr(agent, 'find_crop_suggestion_with_guidance'):
         raise ImportError("Function 'find_crop_suggestion_with_guidance' not found in agent module.")
    if not hasattr(agent, 'retrieve_context'):
         raise ImportError("Function 'retrieve_context' not found in agent module.")
    FARMER_KB = agent.FARMER_KB
    MARKET_KB = agent.MARKET_KB
    print("Agent module and required functions loaded successfully.")
except ImportError as e:
    print(f"Error importing agent module or required functions: {e}")
    FARMER_KB = []
    MARKET_KB = []
    def retrieve_context(query, kb, top_n=3):
        print("Warning: Using dummy retrieve_context.")
        return "Error: KB retrieval agent not found."
    def find_crop_suggestion_with_guidance(farmer_input_data: list):
        print("Warning: Using dummy find_crop_suggestion_with_guidance.")
        return {"error": "Crop suggestion agent not loaded.", "input": dict(zip(['Soil_pH', 'Soil_Moisture', 'Temperature_C', 'Rainfall_mm'], farmer_input_data))}

    class DummyAgent:
        find_crop_suggestion_with_guidance = find_crop_suggestion_with_guidance
        retrieve_context = retrieve_context
        FARMER_KB = []
        MARKET_KB = []
    agent = DummyAgent() 

app = FastAPI(title="Krishi Sahyog")

static_dir = os.path.join(project_root, "static")
templates_dir = os.path.join(project_root, "templates")

if os.path.isdir(static_dir):
     app.mount("/static", StaticFiles(directory=static_dir), name="static")
     print(f"Mounted static directory: {static_dir}")
else:
     print(f"Warning: Static directory not found at expected locations.")

if os.path.isdir(templates_dir):
     templates = Jinja2Templates(directory=templates_dir)
     print(f"Templates directory set to: {templates_dir}")
else:
     print(f"Error: Template directory not found. Exiting.")
     sys.exit(1)

DATABASE_FILE = os.path.join(project_root, "smart_agri.db") 

def init_db():
    """Initializes the SQLite database and creates tables/indexes if they don't exist."""
    try:
        conn = sqlite3.connect(DATABASE_FILE)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS interactions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME NOT NULL,
                query TEXT NOT NULL,
                context_type TEXT NOT NULL, -- 'farmer', 'market', 'crop_suggestion' etc.
                retrieved_context TEXT,      -- Store relevant data (e.g., JSON for suggestions)
                response_preview TEXT        -- Optional preview of AI response
            )
        """)
        print(f"Creating indexes for {DATABASE_FILE} if they don't exist...")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_timestamp ON interactions (timestamp);")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_context_type ON interactions (context_type);")
        conn.commit()
        conn.close()
        print(f"Database '{DATABASE_FILE}' initialized successfully with indexes.")
    except sqlite3.Error as e:
        print(f"Database error during initialization or indexing: {e}")
    except Exception as e:
        print(f"An unexpected error occurred during DB init/indexing: {e}")

def log_interaction(query: str, context_type: str, retrieved_context: str, response_preview: str = ""):
    """Logs an interaction into the SQLite database."""
    try:
        conn = sqlite3.connect(DATABASE_FILE)
        cursor = conn.cursor()
        timestamp = datetime.datetime.now()
        cursor.execute("""
            INSERT INTO interactions (timestamp, query, context_type, retrieved_context, response_preview)
            VALUES (?, ?, ?, ?, ?)
        """, (timestamp, query, context_type, retrieved_context, response_preview))
        conn.commit()
        conn.close()
    except sqlite3.Error as e:
        print(f"Database error during logging: {e}")
    except Exception as e:
        print(f"An unexpected error occurred during interaction logging: {e}")

init_db()
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Serves the home page."""
    return templates.TemplateResponse("home.html", {"request": request})

@app.get("/crop_selection", response_class=HTMLResponse)
async def get_crop_selection_form(request: Request):
    """Displays the form for crop selection input."""
    return templates.TemplateResponse("crop_selection.html", {"request": request})

@app.get("/farmer_help", response_class=HTMLResponse)
async def farmer_help_page(request: Request):
    """Serves the farmer help page."""
    return templates.TemplateResponse("farmer_help.html", {"request": request})

@app.get("/marketing_help", response_class=HTMLResponse)
async def market_help_page(request: Request):
    """Serves the market help page."""
    return templates.TemplateResponse("marketing_help.html", {"request": request})


@app.get("/about", response_class=HTMLResponse)
async def about_page(request: Request):
    """Serves the About Us page."""
    return templates.TemplateResponse("about.html", {"request": request})


@app.post("/suggest_crop", response_class=HTMLResponse)
async def suggest_crop_post(
    request: Request,
    soil_ph: float = Form(...),
    soil_moisture: float = Form(...),
    temperature: float = Form(...),
    rainfall: float = Form(...)
):
    """Processes crop selection form data using the agent and returns suggestions."""
    suggestion_data = None # Initialize
    query_summary = f"Crop Suggestion Query - pH:{soil_ph}, Moist:{soil_moisture}, Temp:{temperature}, Rain:{rainfall}" # Define query early for logging

    try:
        farmer_input = [soil_ph, soil_moisture, temperature, rainfall]

        suggestion_data = agent.find_crop_suggestion_with_guidance(farmer_input)

        context_summary = json.dumps(suggestion_data, default=str) # Convert dict to JSON string
        log_interaction(query=query_summary, context_type='crop_suggestion', retrieved_context=context_summary)

    except Exception as e:
        print(f"Error during crop suggestion processing: {e}")
        suggestion_data = {
            "input": {"Soil_pH": soil_ph, "Soil_Moisture": soil_moisture, "Temperature_C": temperature, "Rainfall_mm": rainfall},
            "found_sustainable": False, 
            "found_alternative": False,
            "error": f"An application error occurred: {e}" 
        }
        log_interaction(query=query_summary, context_type='crop_suggestion_error', retrieved_context=str(e))
        
    template_context = {
        "request": request,
        "suggestion": suggestion_data 
    }

    return templates.TemplateResponse("crop_selection.html", template_context)


def strip_ansi(text: str) -> str:
    ansi_escape = re.compile(r'\x1B\[[0-?]*[ -/]*[@-~]')
    return ansi_escape.sub('', text)

# Define the set of known spinner characters (Braille patterns)
SPINNER_CHARS = set(['⠁', '⠂', '⠄', '⡀', '⢀', '⠠', '⠐', '⠈',
                     '⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏'])

OLLAMA_PATH = "/usr/local/bin/ollama"  

async def stream_query_ollama(prompt: str):
    """Streams the response from the Ollama model character by character."""
    process = None
    try:
        process = await asyncio.create_subprocess_exec(
            OLLAMA_PATH, "run", "llama2:7b-chat", 
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        if process.stdin:
            process.stdin.write(prompt.encode("utf-8"))
            await process.stdin.drain()
            process.stdin.close() 

        while process.stdout:
            char_bytes = await process.stdout.read(1)
            if not char_bytes:
                break 
            try:
                decoded_char = char_bytes.decode("utf-8")
                clean_char = strip_ansi(decoded_char) # Remove ANSI codes
                
                if clean_char and clean_char not in SPINNER_CHARS:
                    yield clean_char 
            except UnicodeDecodeError:
                continue

    except FileNotFoundError:
        error_msg = f"[Error: Ollama executable not found at {OLLAMA_PATH}. Please check the path.]"
        print(error_msg)
        yield error_msg
    except Exception as e:
        error_msg = f"[Error: Could not run Ollama - {e}]"
        print(error_msg)
        yield error_msg
    finally:
        if process:
            if process.stderr:
                try:
                    stderr_output = await process.stderr.read()
                    if stderr_output:
                         print(f"Ollama stderr: {stderr_output.decode(errors='ignore')}")
                except Exception as e_stderr:
                    print(f"Error reading Ollama stderr: {e_stderr}")
            try:
                 if process.returncode is None:
                    process.terminate()
                 await process.wait()
                 print(f"Ollama process finished with code: {process.returncode}")
            except Exception as e_wait:
                 print(f"Error waiting for/terminating Ollama process: {e_wait}")

@app.post("/farmer_help_stream", response_class=StreamingResponse)
async def farmer_help_stream(request: Request, user_question: str = Form(...)):
    """Handles farmer queries using RAG and streams Ollama response."""
    context = agent.retrieve_context(user_question, agent.FARMER_KB, top_n=3) 
    log_interaction(query=user_question, context_type='farmer', retrieved_context=str(context)) 
    prompt = f"""
You are an expert agricultural AI assistant trained on Indian farming data.

Context from dataset:
{context}

Farmer's Query:
"{user_question}"

Based *only* on the provided context and the farmer's query, suggest the best advice, ideal crops, and helpful conditions. If the context seems irrelevant, state that the provided details don't match the query well and offer general advice if possible.
"""
    return StreamingResponse(stream_query_ollama(prompt), media_type="text/plain; charset=utf-8")


@app.post("/marketing_help_stream", response_class=StreamingResponse)
async def marketing_help_stream(request: Request, user_question: str = Form(...)):
    """Handles marketing queries using RAG and streams Ollama response."""
    context = agent.retrieve_context(user_question, agent.MARKET_KB, top_n=3) 
    log_interaction(query=user_question, context_type='market', retrieved_context=str(context)) 
    prompt = f"""
You are an expert agriculture market analyst AI trained on market dynamics and economic factors.

Context from market dataset:
{context}

Marketing Query:
"{user_question}"

Based *only* on the provided context and the marketing query, suggest insights, pricing strategies, demand analysis, or market predictions. If the context seems irrelevant, state that the provided details don't match the query well and offer general advice if possible.
"""
    return StreamingResponse(stream_query_ollama(prompt), media_type="text/plain; charset=utf-8")


if __name__ == "__main__":
    print(f"Starting Uvicorn server...")
    print(f"Access the application at http://127.0.0.1:8000")
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8001, reload=True)
    print("Uvicorn server started successfully.")
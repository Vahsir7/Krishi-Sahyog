import re
import asyncio
import sqlite3 # Import sqlite3
import datetime # Import datetime for timestamps
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

# --- Import the retrieval function and KBs from custom_agent ---
# Ensure custom_agent.py is in the same directory or Python path
try:
    from agents.custom_agent import retrieve_context, FARMER_KB, MARKET_KB
except ImportError:
    print("Error: Could not import from agents.custom_agent. Ensure the file exists and is accessible.")
    # Provide dummy data or exit if essential
    FARMER_KB = []
    MARKET_KB = []
    def retrieve_context(query, kb, top_n=3): return "Error: KB retrieval agent not found."


app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# --- SQLite Database Setup ---
DATABASE_FILE = "smart_agri.db"

def init_db():
    try:
        conn = sqlite3.connect(DATABASE_FILE)
        cursor = conn.cursor()
        # Create table if it doesn't exist
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS interactions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME NOT NULL,
                query TEXT NOT NULL,
                context_type TEXT NOT NULL, -- 'farmer' or 'market'
                retrieved_context TEXT,
                response_preview TEXT
            )
        """)
        # --- Add Indexes ---
        print(f"Creating indexes for {DATABASE_FILE} if they don't exist...")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_timestamp ON interactions (timestamp);")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_context_type ON interactions (context_type);")
        # --- End Add Indexes ---
        conn.commit()
        conn.close()
        print(f"Database '{DATABASE_FILE}' initialized successfully with indexes.")
    except sqlite3.Error as e:
        print(f"Database error during initialization or indexing: {e}")
    except Exception as e:
        print(f"An unexpected error occurred during DB init/indexing: {e}")


def log_interaction(query: str, context_type: str, retrieved_context: str, response_preview: str = ""):
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


# Initialize DB at startup
init_db()
# --- End SQLite Setup ---


# Home page
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("home.html", {"request": request})

# Farmer help page
@app.get("/farmer_help", response_class=HTMLResponse)
async def farmer_help_page(request: Request):
    return templates.TemplateResponse("farmer_help.html", {"request": request}) # Uses streaming JS

# Market help page
@app.get("/marketing_help", response_class=HTMLResponse)
async def market_help_page(request: Request):
    # This now serves the updated marketing_help.html which includes streaming JS
    return templates.TemplateResponse("marketing_help.html", {"request": request})

# Crop selection page (Remains non-streaming for now)
@app.get("/crop_selection", response_class=HTMLResponse)
async def crop_selection_page(request: Request):
    return templates.TemplateResponse("crop_selection.html", {"request": request})


# Helper function to strip ANSI escape sequences
def strip_ansi(text: str) -> str:
    ansi_escape = re.compile(r'\x1B\[[0-?]*[ -/]*[@-~]')
    return ansi_escape.sub('', text)

# Define the set of known spinner characters (Braille patterns)
SPINNER_CHARS = set(['⠁', '⠂', '⠄', '⡀', '⢀', '⠠', '⠐', '⠈',
                     '⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏'])

# Async function to stream Ollama responses character by character (Fixed version)
OLLAMA_PATH = "/usr/local/bin/ollama"  # Adjust as needed

async def stream_query_ollama(prompt: str):
    process = None # Initialize process to None
    try:
        process = await asyncio.create_subprocess_exec(
            OLLAMA_PATH, "run", "llama2:7b-chat", # Or your chosen model
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        # Send the prompt
        if process.stdin:
            process.stdin.write(prompt.encode("utf-8"))
            await process.stdin.drain()
            process.stdin.close()

        # Read one character at a time from stdout
        while process.stdout:
            char_bytes = await process.stdout.read(1)
            if not char_bytes:
                break # End of stream

            try:
                # Decode character, handle potential decoding errors
                decoded_char = char_bytes.decode("utf-8")
                # Strip ANSI codes from the single character
                clean_char = strip_ansi(decoded_char)

                # Yield the clean character immediately if not a spinner
                if clean_char and clean_char not in SPINNER_CHARS:
                    yield clean_char
                    # Optional small delay for smoother visual effect
                    # await asyncio.sleep(0.01) # Uncomment if desired

            except UnicodeDecodeError:
                # Handle cases where a byte doesn't form a complete character
                continue # Skip this byte

    except FileNotFoundError:
        print(f"Error: Ollama executable not found at {OLLAMA_PATH}")
        yield "[Error: Ollama executable not found]"
    except Exception as e:
        print(f"Error during Ollama execution: {e}")
        yield f"[Error: Could not run Ollama - {e}]"
    finally:
        # Ensure stderr is consumed and process is waited for
        if process and process.stderr:
            await process.stderr.read()
        if process:
            try:
                await process.wait()
            except Exception as e:
                 print(f"Error waiting for Ollama process: {e}")


# Streaming response for farmer help
@app.post("/farmer_help_stream", response_class=StreamingResponse)
async def farmer_help_stream(request: Request, user_question: str = Form(...)):
    # Retrieve context using the RAG agent
    context = retrieve_context(user_question, FARMER_KB, top_n=3)
    # Log interaction to SQLite DB
    log_interaction(query=user_question, context_type='farmer', retrieved_context=context)
    prompt = f"""
You are an expert agricultural AI assistant trained on Indian farming data.

Context from dataset:
{context}

Farmer's Query:
"{user_question}"

Based *only* on the provided context and the farmer's query, suggest the best advice, ideal crops, and helpful conditions. If the context seems irrelevant, state that the provided details don't match the query well and offer general advice if possible.
"""
    return StreamingResponse(stream_query_ollama(prompt), media_type="text/plain; charset=utf-8")


# Streaming response for marketing help
@app.post("/marketing_help_stream", response_class=StreamingResponse)
async def marketing_help_stream(request: Request, user_question: str = Form(...)):
    # Retrieve context using the RAG agent
    context = retrieve_context(user_question, MARKET_KB, top_n=3)
    # Log interaction to SQLite DB
    log_interaction(query=user_question, context_type='market', retrieved_context=context)
    prompt = f"""
You are an expert agriculture market analyst AI trained on market dynamics and economic factors.

Context from market dataset:
{context}

Marketing Query:
"{user_question}"

Based *only* on the provided context and the marketing query, suggest insights, pricing strategies, demand analysis, or market predictions. If the context seems irrelevant, state that the provided details don't match the query well and offer general advice if possible.
"""
    return StreamingResponse(stream_query_ollama(prompt), media_type="text/plain; charset=utf-8")


# Placeholder for non-streaming crop selection POST
@app.post("/crop_selection", response_class=HTMLResponse)
async def crop_selection_post(request: Request, user_question: str = Form(...)):
     # TODO: Implement RAG/Ollama call if needed
    context = "Context retrieval not implemented for crop selection yet."
    advice = f"Placeholder advice for crop selection based on: {user_question}\nContext Status: {context}"
    # Log interaction if desired
    log_interaction(query=user_question, context_type='crop_selection', retrieved_context=context)
    return templates.TemplateResponse("crop_selection.html", {"request": request, "user_question": user_question, "advice": advice})
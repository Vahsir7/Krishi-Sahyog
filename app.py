import sys
import os
import re
import asyncio
import sqlite3 # Import sqlite3
import datetime # Import datetime for timestamps
import json # Import JSON for logging structured data
from fastapi import FastAPI, Request, Form, HTTPException # Added HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

# --- Add project root to Python path ---
# Adjust based on your project structure if app.py is nested
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)
# --- End Path Addition ---

# --- Import Agent Functions ---
try:
    import agents.custom_agent as agent # Import the agent module
    # Check if the required functions exist
    if not hasattr(agent, 'find_crop_suggestion_with_guidance'):
         raise ImportError("Function 'find_crop_suggestion_with_guidance' not found in agent module.")
    if not hasattr(agent, 'retrieve_context'):
         raise ImportError("Function 'retrieve_context' not found in agent module.")
    # Make KBs accessible if needed
    FARMER_KB = agent.FARMER_KB
    MARKET_KB = agent.MARKET_KB
    print("Agent module and required functions loaded successfully.")
except ImportError as e:
    print(f"Error importing agent module or required functions: {e}")
    # Provide dummy data/functions if agent fails to load
    FARMER_KB = []
    MARKET_KB = []
    def retrieve_context(query, kb, top_n=3):
        print("Warning: Using dummy retrieve_context.")
        return "Error: KB retrieval agent not found."
    def find_crop_suggestion_with_guidance(farmer_input_data: list):
        print("Warning: Using dummy find_crop_suggestion_with_guidance.")
        return {"error": "Crop suggestion agent not loaded.", "input": dict(zip(['Soil_pH', 'Soil_Moisture', 'Temperature_C', 'Rainfall_mm'], farmer_input_data))}

    # Assign dummy functions if agent failed import
    class DummyAgent:
        find_crop_suggestion_with_guidance = find_crop_suggestion_with_guidance
        retrieve_context = retrieve_context
        FARMER_KB = []
        MARKET_KB = []
    agent = DummyAgent() # Use the dummy agent

# --- Initialize FastAPI App ---
app = FastAPI(title="Smart Agriculture AI")

# --- Setup Static Files & Templates ---
# Adjust paths if your structure is different (e.g., static/templates inside Smart Agri folder)
static_dir = os.path.join(project_root, "static")
templates_dir = os.path.join(project_root, "templates")

if not os.path.isdir(static_dir):
     static_dir = os.path.join(project_root, "Smart Agri", "static")
if not os.path.isdir(templates_dir):
     templates_dir = os.path.join(project_root, "Smart Agri", "templates")

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

# --- SQLite Database Setup ---
DATABASE_FILE = os.path.join(project_root, "smart_agri.db") # Place DB in project root

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

# Initialize DB at startup
init_db()
# --- End SQLite Setup ---


# --- Standard Page Routes ---
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


# --- Crop Suggestion Route (Using New Agent Function) ---
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
        # Prepare input list for the agent function
        farmer_input = [soil_ph, soil_moisture, temperature, rainfall]

        # --- Call the NEW agent function ---
        # Use agent.function_name notation
        suggestion_data = agent.find_crop_suggestion_with_guidance(farmer_input)

        # --- Log interaction ---
        # Log the entire suggestion dictionary as a JSON string for detailed context
        context_summary = json.dumps(suggestion_data, default=str) # Convert dict to JSON string
        log_interaction(query=query_summary, context_type='crop_suggestion', retrieved_context=context_summary)

    except Exception as e:
        print(f"Error during crop suggestion processing: {e}")
        # Create an error structure consistent with agent's return format
        suggestion_data = {
            "input": {"Soil_pH": soil_ph, "Soil_Moisture": soil_moisture, "Temperature_C": temperature, "Rainfall_mm": rainfall},
            "found_sustainable": False, # Ensure these keys exist for template rendering
            "found_alternative": False,
            "error": f"An application error occurred: {e}" # Pass the error message
        }
        # Log the application error itself
        log_interaction(query=query_summary, context_type='crop_suggestion_error', retrieved_context=str(e))
        # Optionally raise HTTPException if you prefer FastAPI's default error handling
        # raise HTTPException(status_code=500, detail="Internal Server Error processing crop suggestion.")


    # --- Prepare context for the template ---
    # The suggestion_data dictionary now directly contains everything the template needs
    # (input, found_sustainable, sustainable_crop, found_alternative, alternative_yield_crop, condition_changes, error)
    template_context = {
        "request": request,
        "suggestion": suggestion_data # Pass the whole dictionary
    }

    # Render the same page, now including the suggestion data (or error)
    return templates.TemplateResponse("crop_selection.html", template_context)


# --- Streaming Logic and Routes (Copied from your provided code) ---

# Helper function to strip ANSI escape sequences
def strip_ansi(text: str) -> str:
    ansi_escape = re.compile(r'\x1B\[[0-?]*[ -/]*[@-~]')
    return ansi_escape.sub('', text)

# Define the set of known spinner characters (Braille patterns)
SPINNER_CHARS = set(['⠁', '⠂', '⠄', '⡀', '⢀', '⠠', '⠐', '⠈',
                     '⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏'])

# Async function to stream Ollama responses
OLLAMA_PATH = "/usr/local/bin/ollama"  # Adjust if your path is different

async def stream_query_ollama(prompt: str):
    """Streams the response from the Ollama model character by character."""
    process = None
    try:
        # Consider adding model selection as a parameter if needed
        process = await asyncio.create_subprocess_exec(
            OLLAMA_PATH, "run", "llama2:7b-chat", # Ensure this model exists
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        # Send the prompt to Ollama
        if process.stdin:
            process.stdin.write(prompt.encode("utf-8"))
            await process.stdin.drain()
            process.stdin.close() # Close stdin to signal end of input

        # Read stdout character by character
        while process.stdout:
            char_bytes = await process.stdout.read(1)
            if not char_bytes:
                break # End of stream
            try:
                decoded_char = char_bytes.decode("utf-8")
                clean_char = strip_ansi(decoded_char) # Remove ANSI codes
                # Filter out spinner characters common in Ollama output
                if clean_char and clean_char not in SPINNER_CHARS:
                    yield clean_char # Yield the clean character
            except UnicodeDecodeError:
                # Skip bytes that don't form a valid UTF-8 character
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
        # Ensure the subprocess is cleaned up properly
        if process:
            if process.stderr:
                try:
                    stderr_output = await process.stderr.read()
                    if stderr_output:
                         print(f"Ollama stderr: {stderr_output.decode(errors='ignore')}")
                except Exception as e_stderr:
                    print(f"Error reading Ollama stderr: {e_stderr}")
            try:
                # Ensure process terminates
                 if process.returncode is None:
                    process.terminate()
                 await process.wait()
                 print(f"Ollama process finished with code: {process.returncode}")
            except Exception as e_wait:
                 print(f"Error waiting for/terminating Ollama process: {e_wait}")


# Streaming response for farmer help
@app.post("/farmer_help_stream", response_class=StreamingResponse)
async def farmer_help_stream(request: Request, user_question: str = Form(...)):
    """Handles farmer queries using RAG and streams Ollama response."""
    context = agent.retrieve_context(user_question, agent.FARMER_KB, top_n=3) # Use agent namespace
    log_interaction(query=user_question, context_type='farmer', retrieved_context=str(context)) # Log context as string
    prompt = f"""
You are an expert agricultural AI assistant trained on Indian farming data.

Context from dataset:
{context}

Farmer's Query:
"{user_question}"

Based *only* on the provided context and the farmer's query, suggest the best advice, ideal crops, and helpful conditions. If the context seems irrelevant, state that the provided details don't match the query well and offer general advice if possible.
"""
    # Return a streaming response from Ollama
    return StreamingResponse(stream_query_ollama(prompt), media_type="text/plain; charset=utf-8")


# Streaming response for marketing help
@app.post("/marketing_help_stream", response_class=StreamingResponse)
async def marketing_help_stream(request: Request, user_question: str = Form(...)):
    """Handles marketing queries using RAG and streams Ollama response."""
    context = agent.retrieve_context(user_question, agent.MARKET_KB, top_n=3) # Use agent namespace
    log_interaction(query=user_question, context_type='market', retrieved_context=str(context)) # Log context as string
    prompt = f"""
You are an expert agriculture market analyst AI trained on market dynamics and economic factors.

Context from market dataset:
{context}

Marketing Query:
"{user_question}"

Based *only* on the provided context and the marketing query, suggest insights, pricing strategies, demand analysis, or market predictions. If the context seems irrelevant, state that the provided details don't match the query well and offer general advice if possible.
"""
    # Return a streaming response from Ollama
    return StreamingResponse(stream_query_ollama(prompt), media_type="text/plain; charset=utf-8")


# --- Removed Redundant Placeholder Route ---
# @app.post("/crop_selection", ...) was removed as /suggest_crop handles this now.


# --- Run the app (for development) ---
if __name__ == "__main__":
    # Optional: Add checks here to verify agent data loading status if needed
    # try:
    #     if agent.FARMER_DF.empty or agent.nn_model is None: # Example check
    #         print("\n*** Warning: Agent data/model may not be loaded. Crop suggestions might fail. ***\n")
    # except AttributeError:
    #      print("\n*** Warning: Could not check agent data status. ***\n")

    print(f"Starting Uvicorn server...")
    print(f"Access the application at http://127.0.0.1:8000")
    # Use uvicorn.run for programmatic execution
    import uvicorn
    # reload=True is helpful for development
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
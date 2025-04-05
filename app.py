import re
import asyncio
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

# Assuming nlp_advisor still provides context generators
from nlp_advisor import generate_context_for_crop, generate_context_for_market

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Home page
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("home.html", {"request": request})

# Farmer help page
@app.get("/farmer_help", response_class=HTMLResponse)
async def farmer_help_page(request: Request):
    return templates.TemplateResponse("farmer_help.html", {"request": request})

# Market help page
@app.get("/marketing_help", response_class=HTMLResponse)
async def market_help_page(request: Request):
    # Assuming you want a streaming version for market help too
    return templates.TemplateResponse("marketing_help_stream.html", {"request": request}) # Changed template if streaming market help

# Crop selection page (Assuming this doesn't need streaming for now)
@app.get("/crop_selection", response_class=HTMLResponse)
async def crop_selection_page(request: Request):
    return templates.TemplateResponse("crop_selection.html", {"request": request})


# Helper function to strip ANSI escape sequences
def strip_ansi(text: str) -> str:
    ansi_escape = re.compile(r'\x1B\[[0-?]*[ -/]*[@-~]')
    return ansi_escape.sub('', text)

# Define the set of known spinner characters (Braille patterns)
# Add more if you see different ones appearing
SPINNER_CHARS = set(['⠁', '⠂', '⠄', '⡀', '⢀', '⠠', '⠐', '⠈', # Dots
                     '⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏' # Spinners
                    ])

# Async function to stream Ollama responses character by character (from custom_agent.py)
OLLAMA_PATH = "/usr/local/bin/ollama"  # Adjust as needed

async def stream_query_ollama(prompt: str):
    process = await asyncio.create_subprocess_exec(
        OLLAMA_PATH, "run", "llama2:7b-chat",
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )

    # Send the prompt
    process.stdin.write(prompt.encode("utf-8"))
    await process.stdin.drain()
    process.stdin.close()

    # Read one character at a time from stdout
    while True:
        char_bytes = await process.stdout.read(1) # Read byte by byte
        if not char_bytes:
            break # End of stream

        try:
            # Decode character, handle potential decoding errors
            decoded_char = char_bytes.decode("utf-8")
            # Strip ANSI codes from the single character
            clean_char = strip_ansi(decoded_char)

            # --- Add check for spinner characters ---
            if clean_char and clean_char not in SPINNER_CHARS:
                 yield clean_char
                 # Optional small delay for smoother visual effect
                 await asyncio.sleep(0.01) # Adjust delay as needed

        except UnicodeDecodeError:
            # Handle cases where a byte doesn't form a complete character
            continue

    # Ensure stderr is consumed but not yielded (unless debugging)
    await process.stderr.read()
    await process.wait() # Wait for the process to fully finish

# Streaming response for farmer help
@app.post("/farmer_help_stream", response_class=StreamingResponse)
async def farmer_help_stream(request: Request, user_question: str = Form(...)):
    context = generate_context_for_crop() # From nlp_advisor.py
    prompt = f"""
You are an expert agricultural AI assistant trained on Indian farming data.

Context from dataset:
{context}

Farmer's Query:
"{user_question}"

Using the above data, suggest the best advice, ideal crops, and helpful conditions. Keep the response concise and clear.
"""
    # Use the character-by-character streaming function
    return StreamingResponse(stream_query_ollama(prompt), media_type="text/plain")


# Streaming response for marketing help
@app.post("/marketing_help_stream", response_class=StreamingResponse)
async def marketing_help_stream(request: Request, user_question: str = Form(...)):
    context = generate_context_for_market() # From nlp_advisor.py
    prompt = f"""
You are an expert agriculture market analyst AI trained on market dynamics and economic factors.

Context from market dataset:
{context}

Marketing Query:
"{user_question}"

Using the above data, suggest insights, best pricing strategies, demand analysis, or market predictions. Keep the response concise and clear.
"""
     # Use the character-by-character streaming function
    return StreamingResponse(stream_query_ollama(prompt), media_type="text/plain")
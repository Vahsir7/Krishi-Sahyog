# app.py
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from nlp_advisor import ask_ollama_crop_advisor, ask_ollama_market_advisor

app = FastAPI()

# Mount static files if you have any (like CSS)
app.mount("/static", StaticFiles(directory="static"), name="static")

templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("base.html", {"request": request})

# Farmer help page using the NLP crop advisor
@app.get("/farmer_help", response_class=HTMLResponse)
async def farmer_help(request: Request):
    return templates.TemplateResponse("farmer_help.html", {"request": request})

@app.post("/farmer_help", response_class=HTMLResponse)
async def farmer_help_post(request: Request, user_question: str = Form(...)):
    advice = ask_ollama_crop_advisor(user_question)
    return templates.TemplateResponse("farmer_help.html", {"request": request, "advice": advice, "user_question": user_question})

# Market help page using the NLP market advisor
@app.get("/marketing_help", response_class=HTMLResponse)
async def marketing_help(request: Request):
    return templates.TemplateResponse("marketing_help.html", {"request": request})

@app.post("/marketing_help", response_class=HTMLResponse)
async def marketing_help_post(request: Request, user_question: str = Form(...)):
    advice = ask_ollama_market_advisor(user_question)
    return templates.TemplateResponse("marketing_help.html", {"request": request, "advice": advice, "user_question": user_question})

# Optionally, a crop selection page that also uses NLP for more detailed advice.
@app.get("/crop_selection", response_class=HTMLResponse)
async def crop_selection(request: Request):
    return templates.TemplateResponse("crop_selection.html", {"request": request})

@app.post("/crop_selection", response_class=HTMLResponse)
async def crop_selection_post(request: Request, user_question: str = Form(...)):
    # Here you could combine numeric field inputs with NLP reasoning
    advice = ask_ollama_crop_advisor(user_question)
    return templates.TemplateResponse("crop_selection.html", {"request": request, "advice": advice, "user_question": user_question})

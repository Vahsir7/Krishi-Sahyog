from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from agents.custom_agent import get_farmer_advice, get_market_advice

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("base.html", {"request": request})

# Farmer Help Page
@app.get("/farmer_help", response_class=HTMLResponse)
async def farmer_help(request: Request):
    return templates.TemplateResponse("farmer_help.html", {"request": request})

@app.post("/farmer_help", response_class=HTMLResponse)
async def farmer_help_post(request: Request, user_question: str = Form(...)):
    print(f"User question: {user_question}")
    advice = get_farmer_advice(user_question)
    
    return templates.TemplateResponse("farmer_help.html", {
        "request": request,
        "advice": advice,
        "user_question": user_question
    })

# Market Help Page
@app.get("/marketing_help", response_class=HTMLResponse)
async def marketing_help(request: Request):
    return templates.TemplateResponse("marketing_help.html", {"request": request})

@app.post("/marketing_help", response_class=HTMLResponse)
async def marketing_help_post(request: Request, user_question: str = Form(...)):
    advice = get_market_advice(user_question)
    return templates.TemplateResponse("marketing_help.html", {
        "request": request,
        "advice": advice,
        "user_question": user_question
    })

# Crop Selection Page (using farmer advice logic)
@app.get("/crop_selection", response_class=HTMLResponse)
async def crop_selection(request: Request):
    return templates.TemplateResponse("crop_selection.html", {"request": request})

@app.post("/crop_selection", response_class=HTMLResponse)
async def crop_selection_post(request: Request, user_question: str = Form(...)):
    advice = get_farmer_advice(user_question)
    return templates.TemplateResponse("crop_selection.html", {
        "request": request,
        "advice": advice,
        "user_question": user_question
    })

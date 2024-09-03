from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles

app = FastAPI()

# Serve static files (e.g., images)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Setup Jinja2 templates
templates = Jinja2Templates(directory="templates")

# Dummy functions for extracting entities and sentiment analysis
def extract_entities(sentence: str) -> dict:
    # Replace with your actual entity extraction function
    return {"Entity1": "Value1", "Entity2": "Value2"}

def sentiment_analysis(sentence: str) -> list:
    # Replace with your actual sentiment analysis function
    return [{"polarity": 0.5, "subjectivity": 0.6}]

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/process_sentence", response_class=HTMLResponse)
async def process_sentence(request: Request, sentence: str = Form(...)):
    response = {
        "Sentence": sentence,
        "Entities": extract_entities(sentence),
        "Sentiment Analysis Results": sentiment_analysis(sentence)
    }
    return templates.TemplateResponse("index.html", {"request": request, "response": response})


import threading
from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from mainn import my_main
#import thread2



app = FastAPI()

# Serve static files (e.g., images)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Setup Jinja2 templates
templates = Jinja2Templates(directory="templates")



@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/process_sentence", response_class=HTMLResponse)
async def process_sentence(request: Request, sentence: str = Form(...)):

    data = my_main(sentence)
    
    
    # Prepare the response data
    def clean_format(text):
        return text.replace("[", "").replace("]", "").replace("'", "")

    # Apply the clean_format function to both entity and sentiment lists
    cleaned_entity_list = [clean_format(entity) for entity in data["entities"]]
    cleaned_sentiment_list = [clean_format(sentiment) for sentiment in data["sentiment"]]


    response_data = {
        "Sentence": sentence,
        "Entities": cleaned_entity_list,
        "Sentiments": cleaned_sentiment_list
    }

    #thread2.add_sentences(sentence,)

    return templates.TemplateResponse("index.html", {"request": request, "response": response_data})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)

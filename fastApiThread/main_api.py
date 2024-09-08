from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from main import my_main, uploadFile
import queue
import threading
import json
import shutil
import time
import os
import uvicorn

app = FastAPI()

# Serve static files (e.g., images)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Setup Jinja2 templates
templates = Jinja2Templates(directory="templates")

# Initialize task queue
task_queue = queue.Queue()
processed_sentences = []
response_data = {}
lock = threading.Lock()  # For thread-safe access to shared data


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/process_sentence", response_class=HTMLResponse)
async def process_sentence(request: Request, sentence: str = Form(...)):
    # Add the received sentence to the queue

    data = my_main(sentence)
    task_queue.put(sentence)

    # Optionally process the sentence and return a response

    def clean_format(text):
        return text.replace("[", "").replace("]", "").replace("'", "")

    cleaned_entity_list = [clean_format(entity) for entity in data["entities"]]
    cleaned_sentiment_list = [clean_format(sentiment) for sentiment in data["sentiment"]]

    global response_data
    response_data = {
        "Sentence": sentence,
        "Entities": cleaned_entity_list,
        "Sentiments": cleaned_sentiment_list
    }

    # uploadFile()
    return templates.TemplateResponse("index.html", {"request": request, "response": response_data})

    # Function to simulate sentence processing (takes 10 seconds per sentence)


def process_and_save_sentence_data(sentence):
    return my_main(sentence)


# Function to save data to JSON file
def save_results(filename, data):
    with open(filename, 'w', encoding='utf-8') as file:
        json.dump(data, file, indent=4, ensure_ascii=False)


# Function to read JSON file
def read_json_file(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        return data
    except Exception as e:
        print(f"Error reading the JSON file: {e}")
        return None


# Function to empty a JSON file (reset its contents)
def empty_json_file(file_path):
    try:
        with open(file_path, 'w', encoding='utf-8') as file:
            json.dump({"sentences": []}, file)
        print(f"File '{file_path}' has been emptied.")
    except Exception as e:
        print(f"Error emptying the JSON file: {e}")


# Function to copy the JSON file to a new path
def send_json_file(source_path, destination_path):
    try:
        shutil.copy(source_path, destination_path)
        print(f"File copied to {destination_path}.")
    except Exception as e:
        print(f"Error copying the JSON file: {e}")


# Function to process the queue and store results in processed_sentences
def process_queue():
    while True:
        if not task_queue.empty():
            sentence = task_queue.get()
            processed_sentence = response_data
            with lock:
                processed_sentences.append(processed_sentence)
            print(f"Processed and saved: {processed_sentence}")
        time.sleep(1)  # Avoid excessive CPU usage


# Function to handle time-based operations (every 55 seconds)
def time_based_operations():
    json_file = "data.json"
    destination_path = "results\data.json"

    while True:
        current_time = int(time.time())

        if current_time % 55 == 0:
            with lock:
                if processed_sentences:
                    print("55-second mark reached, sending processed sentences...")
                    save_results(json_file, processed_sentences)
                    send_json_file(json_file, destination_path)
                    uploadFile()
                    empty_json_file(json_file)
                    processed_sentences.clear()
                    print("Processed sentences sent and cleared.")
                else:
                    print("55-second mark reached, but no processed sentences to send.")

        time.sleep(1)  # Avoid excessive CPU usage


# Run threads after the application has started
def run_threads():
    queue_thread = threading.Thread(target=process_queue)
    time_thread = threading.Thread(target=time_based_operations)

    queue_thread.start()
    time_thread.start()

    queue_thread.join()
    time_thread.join()


if __name__ == "__main__":
    import threading
    import uvicorn

    # Start the server
    server_thread = threading.Thread(target=lambda: uvicorn.run(app, host="127.0.0.1", port=8000))
    server_thread.start()

    # Run background threads
    run_threads()

    server_thread.join()
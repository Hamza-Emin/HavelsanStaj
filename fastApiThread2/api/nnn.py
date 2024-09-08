import queue
import time
import json
import shutil
import threading
import asyncio
from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from mainn import my_main  # Assuming my_main is the processing function

# Setup FastAPI
app = FastAPI()

# Serve static files (e.g., images)
app.mount("/static", StaticFiles(directory=r"C:\Users\Hamza\Desktop\dilarafastthreadbirleşimi\api\static"), name="static")

# Setup Jinja2 templates
templates = Jinja2Templates(directory=r"C:\Users\Hamza\Desktop\dilarafastthreadbirleşimi\api\templates")

# Queue and threading logic
task_queue = queue.Queue()
processed_sentences = []
lock = threading.Lock()  # For thread-safe access to shared data

# Function to simulate sentence processing
def process_and_save_sentence_data(sentence):
    """Process the sentence using the my_main function."""
    return f"Processed: {my_main(sentence)}"

# Function to save data to JSON file
def save_results(filename, data):
    """Append data to a JSON file."""
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)

# Function to read JSON file
def read_json_file(file_path):
    """Reads a JSON file and returns its contents."""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        return data
    except Exception as e:
        print(f"Error reading the JSON file: {e}")
        return None

# Function to empty a JSON file (reset its contents)
def empty_json_file(file_path):
    """Empties the JSON file by resetting its contents."""
    try:
        with open(file_path, 'w', encoding='utf-8') as file:
            json.dump({"sentences": []}, file)
        print(f"File '{file_path}' has been emptied.")
    except Exception as e:
        print(f"Error emptying the JSON file: {e}")

# Function to copy the JSON file to a new path
def send_json_file(source_path, destination_path):
    """Copies the JSON file from source_path to destination_path."""
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
            processed_sentence = process_and_save_sentence_data(sentence)
            with lock:
                processed_sentences.append(processed_sentence)
            print(f"Processed and saved: {processed_sentence}")
        time.sleep(1)  # Avoid excessive CPU usage

# Function to handle time-based operations (every 55 seconds)
def time_based_operations():
    json_file = "data.json"
    destination_path = "results/data.json"
    
    while True:
        current_time = int(time.time())
        if current_time % 55 == 0:
            with lock:
                if processed_sentences:
                    print("55-second mark reached, sending processed sentences...")
                    save_results(json_file, processed_sentences)
                    send_json_file(json_file, destination_path)
                    empty_json_file(json_file)
                    processed_sentences.clear()  # Clear after sending
                    print("Processed sentences sent and cleared.")
                else:
                    print("55-second mark reached, but no processed sentences to send.")
        time.sleep(1)

# Start threads for background processing
queue_thread = threading.Thread(target=process_queue)
time_thread = threading.Thread(target=time_based_operations)

# Start the threads
queue_thread.start()
time_thread.start()

# Asynchronous wrapper for my_main
async def run_mymain_async(sentence):
    loop = asyncio.get_running_loop()
    # Run the blocking function in an executor
    processed_sentence = await loop.run_in_executor(None, my_main, sentence)
    return processed_sentence

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/process_sentence", response_class=HTMLResponse)
async def process_sentence(request: Request, sentence: str = Form(...)):
    # Process the sentence
    task_queue.put(sentence)
    print(f"Added to queue: {sentence}")

    processed_sentence = await run_mymain_async(sentence)
    print("Response data:", processed_sentence)

    # Construct response data
    response_data = {
        "Sentence": sentence,
        "Entities": ['vodafone -> Organization', 'turkcell -> Organization'],  # Example, replace with actual result
        "Sentiment Analysis Results": ['vodafone -> Olumlu', 'turkcell -> Olumsuz']  # Example, replace with actual result
    }

    # Ensure that the response data is serializable
    safe_response_data = {}
    for k, v in response_data.items():
        if isinstance(v, (str, int, float, list, dict)):
            safe_response_data[k] = v
        else:
            safe_response_data[k] = str(v)  # Convert non-serializable data to string

    return templates.TemplateResponse("index.html", {"request": request, "response": safe_response_data})


# FastAPI app runner
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)

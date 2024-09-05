import queue
import time
import json
import shutil
import threading
from main import my_main

task_queue = queue.Queue()
processed_sentences = []
lock = threading.Lock()  # For thread-safe access to shared data

# Function to simulate sentence processing (takes 10 seconds per sentence)
def process_and_save_sentence_data(sentence):
    """Simulate processing time (10 seconds) and return processed sentence."""
    return f"Processed: {my_main(sentence)}"

# Function to save data to JSON file
def save_results(filename, data):
    """Append data to a JSON file."""
    with open(filename, 'w', encoding='utf-8') as f:
        # Wrap the list of sentences in a dictionary for proper JSON structure
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
            json.dump({"sentences": []}, file)  # Start with an empty "sentences" array
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

        # Check if it's time to perform the 55-second operation
        if current_time % 55 == 0:
            with lock:
                if processed_sentences:  # Only proceed if there are processed sentences
                    print("55-second mark reached, sending processed sentences...")
                    save_results(json_file, processed_sentences)
                    send_json_file(json_file, destination_path)
                    empty_json_file(json_file)
                    processed_sentences.clear()  # Clear after sending
                    print("Processed sentences sent and cleared.")
                else:
                    print("55-second mark reached, but no processed sentences to send.")

        time.sleep(1)  # Avoid excessive CPU usage

# Adding sentences to the queue (run this in a separate thread)
def add_sentences():
    sentences = [
        "Turk Telekom her yönden çok daha iyi. Vodafone kullanıyorum ve 70 TL paket ücreti ödüyorum, 15gb 1000dk veriyor. Pahalı geliyor. Turkcell konusunda ise emin değilim.",
        "Turkcell çok iyi ama Vodafone kötü.",
        "turkcell çok pahalı. aynı koşullardaki tarifelere bakınca mecburen vodafone tercih ediyorum",
        "Each sentence will be handled and saved.",
        "Check the interval when sentences are processed.",
        "This is the sixth sentence in the list.",
        "Testing with various sentences to ensure functionality.",
        "Sentences should be processed and saved.",
        "Look at how the script manages the workload.",
        "Finally, this is the tenth sentence for the test."
    ]
    
    for sentence in sentences:
        task_queue.put(sentence)
        print(f"Added to queue: {sentence}")
        time.sleep(1)  # Simulate time gap between sentences being added

# Create and start threads for concurrent execution
queue_thread = threading.Thread(target=process_queue)
time_thread = threading.Thread(target=time_based_operations)
sentence_thread = threading.Thread(target=add_sentences)

queue_thread.start()
time_thread.start()
sentence_thread.start()

# Join threads to the main thread so they continue running
queue_thread.join()
time_thread.join()
sentence_thread.join()
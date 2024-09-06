import queue
import time
import json
import shutil
import threading
from main import my_main
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
import os

# Google API erişimi için gerekli kapsam (scope)
SCOPES = ['https://www.googleapis.com/auth/drive.file']

# credentials.json dosyasının bilgisayarınızdaki tam yolu
creds_path = "C:/Users/Alperen/Desktop/main/credentials.json"

# Kimlik doğrulama ve token yenileme
creds = None
if os.path.exists('token.json'):
    creds = Credentials.from_authorized_user_file('token.json', SCOPES)
if not creds or not creds.valid:
    if creds and creds.expired and creds.refresh_token:
        creds.refresh(Request())
    else:
        flow = InstalledAppFlow.from_client_secrets_file(creds_path, SCOPES)
        creds = flow.run_local_server(port=0)
    with open('token.json', 'w') as token:
        token.write(creds.to_json())

# Google Drive API'yi başlatma
service = build('drive', 'v3', credentials=creds)

# Yüklenecek dosyanın yolu (bilgisayarınızda)
file_path = "C:/Users/Alperen/Desktop/main/data.json"

# Dosyanın Drive'daki adını belirleyelim
file_name = os.path.basename(file_path)

# Aynı ada sahip bir dosya olup olmadığını kontrol et
results = service.files().list(q=f"name='{file_name}'", fields="files(id, name)").execute()
items = results.get('files', [])

def updateFile():
    if items:
        # Dosya varsa güncelle
        file_id = items[0]['id']
        media = MediaFileUpload(file_path, mimetype='text/plain')
        updated_file = service.files().update(fileId=file_id, media_body=media).execute()
        print(f"Dosya başarıyla güncellendi. Dosya ID'si: {updated_file.get('id')}")
    else:
        # Dosya yoksa oluştur
        file_metadata = {'name': file_name}
        media = MediaFileUpload(file_path, mimetype='text/plain')
        created_file = service.files().create(body=file_metadata, media_body=media, fields='id').execute()
        print(f"Dosya başarıyla yüklendi. Dosya ID'si: {created_file.get('id')}")


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
                    updateFile()
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
        "red paketim rezalet o yüzden turkcelle geçiş yapacağım. twitch izlerken sürekli donma yaşıyorum.. turkcell müşteri hizmetleri çok daha iyi",
        "Turk Telekom kullanmaya başladım başlayalı gençleştim resmen. Vodafone kullanırken kelimenin tam anlamıyla yaşlanmıştım.",
        "hepsi halkı soymaya,cebinden haksızca para kazanmaya yemin etmiş sanki. al birini vur diğerine zehir zıkkım olsun. vodafone",
        "bende sıkıntı yok valla turkcell pişmanlıktır. vodafone candır.",
        "al bende o kadar ankara’nın göbeğinde çekmiyor. bide kalite diyene ayar oluyorum. fazla faturalarda cabası. vodafone",
        "vodafone inanılmaz pahalı. paketim bitmesine 1 gün kala konuşma dk’kam bitti. anında 24 tl yapıştırıp, 1 günlüğüne 250 dakika verdiler. hemde hiç sormuyorlar. şikayet hakkım bile yokmuş. yılım dolduğu an turktelekom ‘a geçmeyi düşünüyorum, bakalım.",
        "turkcell kazık. türk telekom ekonomik ama çoğu yerde çekmiyor. i̇kisinin ortası vodafone. çekimide öyle bazılarının dediği gibi kötü değil"
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


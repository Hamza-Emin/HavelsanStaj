#!pip install https://huggingface.co/turkish-nlp-suite/tr_core_news_trf/resolve/main/tr_core_news_trf-1.0-py3-none-any.whl

import ast
import spacy
import json
import re
import tensorflow as tf
from transformers import BertTokenizer, TFBertForSequenceClassification
from spacy import displacy

nlp = spacy.load('tr_core_news_trf')
load_path = 'spacy_trained_model' 
nlp.add_pipe('sentencizer')

#sentence = "Turk Telekom her yönden çok daha iyi. Vodafone kullanıyorum ve 70 TL paket ücreti ödüyorum, 15gb 1000dk veriyor. Pahalı geliyor. Turkcell konusunda ise emin değilim."
#sentence = "Turkcell'den çok memnun kaldım ama Vodafone için aynı şeyleri söyleyemem. Hiç memnun kalmadım.""
#sentence = "al birini vur ötekine. kurumsal dolandırıcılardan bıktım @turkcell @vodafonetr"
#sentence = "turkcell çok pahalı. aynı koşullardaki tarifelere bakınca mecburen vodafone tercih ediyorum"
#sentence = "turkcell kazık. türk telekom ekonomik ama çoğu yerde çekmiyor. i̇kisinin ortası vodafone. çekimide öyle bazılarının dediği gibi kötü değil"
#sentence = "vodafone inanılmaz pahalı. paketim bitmesine 1 gün kala konuşma dk’kam bitti. anında 24 tl yapıştırıp, 1 günlüğüne 250 dakika verdiler. hemde hiç sormuyorlar. şikayet hakkım bile yokmuş.  yılım dolduğu an turktelekom ‘a geçmeyi düşünüyorum, bakalım."
#sentence = "al bende o kadar ankara’nın göbeğinde çekmiyor. bide kalite diyene ayar oluyorum. fazla faturalarda cabası. vodafone"
#sentence = "bende sıkıntı yok valla turkcell pişmanlıktır. vodafone candır."
#sentence = "hepsi halkı soymaya,cebinden haksızca para kazanmaya yemin etmiş sanki. al birini vur diğerine zehir zıkkım olsun. vodafone;0"
#sentence = "Turk Telekom kullanmaya başladım başlayalı gençleştim resmen. Vodafone kullanırken kelimenin tam anlamıyla yaşlanmıştım."

def ab_model(text): 
    nlp = spacy.load(load_path) 
    result = []
    doc = nlp(text)
    for ent in doc.ents:
        result.append(ent.text)

    return result


def ab_model_with_label(text): 
    nlp = spacy.load(load_path) 
    result = []
    doc = nlp(text)
    for ent in doc.ents:
        result.append(ent.label_)

    return result


def get_split_sentences_with_conjunction(entities1, text):
    nlp = spacy.load('tr_core_news_trf')
    doc = nlp(text) # metnin tokenize haline getirilmesi
    split_sentences = [] # çıkarılan cümlelerin tutulacağı dizinin başlatılması
    add_old_entity = False # ilk durumda eski entity eklenmesi bool değeri False verilir.

    for token in doc: # metin içindeki her bir kelime için:
      sentence = [] # elde edilen metnin tutulacağ dizinin başlatılması
      entities = [token for token in doc if token.text in entities1] # entity lerin metin içinden tanınması
      if token.is_sent_end and token in entities: # token en sonda ise ve bir entity ise:
        entity = doc[token.i] # entity metin içinden alınır ve bir değişkende tutulur
        text = doc[:token.i].text.strip() # sondaki entityden önceki bütün metin bir değişkende tutulur
        sentence.append(entity.text + "  " + text) # başa sondaki entity gelecek şekilde bütün metin entity nin sonunda eklenir
        return sentence # son metnin tutulduğu dizi döndürülür
     
    for sent in doc.sents: # metnin her bir cümlesi için:
        conjunctions = [token for token in sent if token.pos_ in ['CCONJ', 'SCONJ'] or token.text == "\0"] # bağlaçların metin içinden tanınması
        entities = [token for token in sent if token.text in entities1] # entity lerin metin içinden tanınması
        k = 0 # ilk bağlaç için bir sayaç oluşturulması
        if conjunctions: # metinde bağlaç var ise:
          for conjunction in conjunctions: # her bir bağlaç için:
            try:
              print(conjunction.text)
              if k==0: # ilk bağlaç için:
                  if sent[conjunction.i-1] in entities and sent[conjunction.i+1] in entities: # eğer bağlaçtan bir önceki ve sonraki kelime bir entity ise:
                      old_entity = sent[conjunction.i-1] # önceki entity bir değişkende tutulur
                      add_old_entity = True # eski entity sonradan ekleneceği için bool değişkeni True yapılır
                  else: # eğer bağlaçtan bir önceki ve sonraki kelime bir entity değil ise:
                      text = sent[:conjunction.i].text.strip() # bağlaçtan önceki cümle metinden alınır
                      split_sentences.append(text) # alınan metin diziye eklenir
                  k += 1 # ilk bağlaçtan çıkıldığı için sayaç arttırılır
                  old_conjunction = conjunction # eski bağlaç bir değişkende tutulur
              else: # ilk bağlaçtan sonraki bağlaçlar ise:
                  if sent[conjunction.i-1] in entities and sent[conjunction.i+1] in entities: # eğer bağlaçtan bir önceki ve sonraki kelime bir entity ise:
                      old_entity = sent[conjunction.i-1] # önceki entity bir değişkende tutulur
                      add_old_entity = True # eski entity sonradan ekleneceği için bool değişkeni True yapılır
                  else: # eğer bağlaçtan bir önceki ve sonraki kelime bir entity değil ise:
                      text = sent[old_conjunction.i+1:conjunction.i].text.strip() # eski bağlaçtan şu anki bağlaça kadar olan cümle metinden alınır
                      split_sentences.append(text) # alınan metin diziye eklenir
                      if add_old_entity == True: # eski entity nin eklenmesi gerekiyor ise:
                          text = sent[old_entity.i].text.strip() # eski entity cümleden alınır
                          text2 = sent[old_conjunction.i+2:conjunction.i].text.strip() # eski bağlaçtan ve entityden sonra şu anki bağlaça kadar olan cümle metinden alınır
                          split_sentences.append(text + " " + text2) # eski entity ile alınan cümle arasına bir boşluk eklenerek diziye eklenir
                          add_old_entity = False # eski entity alındığı için bool False değerine getirilir
                  old_conjunction = conjunction # eski bağlaç bir değişkende tutulur
                  k += 1 # baglac sayacı bir arttırılır
            except IndexError:
                print("IndexError: [E1002] Span index out of range.")
                split_sentences.append(sent.text)
          split_sentences.append(sent[conjunction.i + 1:].text.strip()) # son baglactan sonra kalan metin alınır
        else:
            split_sentences.append(sent.text)
    while("" in split_sentences): # son alınan metin boş string ise diziden silinir
      split_sentences.remove("")

    return split_sentences


# Kaç tane entity olduğunu sayar
def howmanyentity(sentence, entity):
    count = 0

    words = sentence.split(" ")
    for i in words:
        if i in entity:
            count = count+1

    return count


fiilimsi_suffixes = [
    "ken", "alı", "eli", "madan", "meden", "ince", "ınca", "unca", "ünce",
    "ıp", "up", "üp", "arak", "erek", "dıkça", "dikçe", "dukça", "dükçe",
    "tıkça", "tikçe", "tukça", "tükçe", "a", "e", "r", "maz", "mez",
    "casına", "cesine", "meksizin", "maksızın", "dığında", "diğinde",
    "duğunda", "düğünde", "tığında", "tiğinde", "tuğunda", "tüğünde"
]

def find_fiilimsi(sentence):
    doc = nlp(sentence)

    fiilimsis = []

    for token in doc:
        # Check if the token ends with any of the fiilimsi suffixes
        if any(token.text.endswith(suffix) for suffix in fiilimsi_suffixes):
            # Check dependency and POS tags to filter fiilimsi
            if token.dep_ in ["advcl", "ccomp", "amod", "acl"] or (token.pos_ in ['VERB', 'AUX'] and token.dep_ not in ["ROOT"]):
                fiilimsis.append(token.text)

    return fiilimsis


def clean_text_for_dependency_parsing(sentence):
    import string
    keep_punctuation = {'.', '?', ','}

    cleaned_sentence = ''.join(char for char in sentence if char not in string.punctuation or char in keep_punctuation)
    cleaned_sentence = cleaned_sentence.replace(",", " ")
    return cleaned_sentence


def normalize_entities(entities, sentence):
    for entity in entities:
        pattern = re.compile(rf'\b{entity}(\w*\'?\w*)?\b', re.IGNORECASE)
        sentence = pattern.sub(entity, sentence)
    return sentence


def remove_spaces_from_entities(entities, sentence):
    for entity in entities:
        if " " in entity: 
            position = sentence.find(entity)     
            if position != -1:     
                entity_no_spaces = entity.replace(" ", "")     
                offset = 0  
                for i, char in enumerate(entity):
                    if char == " ":         
                        newpos = position + i - offset
                        sentence = sentence[:newpos] + sentence[ newpos + 1:]
                        offset += 1 
                entities[entities.index(entity)] = entity_no_spaces

    return sentence, entities
   

def splitting(entity : list, sentence: str):
    copy_entity = entity.copy()

    sentence = clean_text_for_dependency_parsing(sentence)
    sentence = normalize_entities(entity, sentence)
 
    sentence , entity = remove_spaces_from_entities(entity,sentence)
    sentenceresults = list()
    if sentence.endswith("."):
        sentence = sentence[0:-1]

    numberofentities=howmanyentity(sentence=sentence,entity=entity) #change this function to handle empty spaces in the end
    #print(f"numberofentities of this function is : {numberofentities}")
    if ( numberofentities == 1): # if there is only 1 entity
        sentenceresults.append(sentence)

    elif(numberofentities >1 ):
        numberoffiilimsi = len(find_fiilimsi(sentence=sentence))
        #print(f"Number of fiilimsi is :{numberoffiilimsi}")

        if(numberoffiilimsi==1) :
            #print("Hamza")
            entitiypositions = []
            for i in entity:
                entitiypositions.append(sentence.index(i))

            fiilimsiposition = sentence.find(find_fiilimsi(sentence=sentence)[0])
            #print("Emin")
            whereisfiilimsi = all(index < fiilimsiposition for index in entitiypositions)
            if whereisfiilimsi:
                for i in entitiypositions:
                    entitylen = len(sentence[i:].split()[0])
                    #print(f" Empty len is {entitylen}")
                    #print(f"entitiyposition: {entitylen}, fiilimsipositions: {fiilimsiposition}")
                    if i == 0:
                        temp_sentence = sentence[:entitylen] + " " + sentence[fiilimsiposition:]
                    if i !=0:
                        temp_sentence = sentence[i:i+entitylen]+ " " + sentence[fiilimsiposition:]
                    sentenceresults.append(temp_sentence)
            else : # that means there is a fiilimsi between entities. there can be entity fiilimsi entity , entity entity fiilimsi entity
                beforeentitiesposition = [] # BEFORE THE FİİLİMSİ ENTİTİES
                afterentitiesposition = [] # AFTER THE FİİLİMSİ ENTİTİES
                fiilimsiposition = sentence.find(find_fiilimsi(sentence=sentence)[0]) # FİİLİMSİ POİZİTON

                for i in entity:
                    if sentence.find(i) < fiilimsiposition:
                        beforeentitiesposition.append(sentence.find(i))
                    elif sentence.find(i) > fiilimsiposition:
                        afterentitiesposition.append(sentence.find(i))

                #handle before entities
                #handle after entities
                #append the result
                for positions in beforeentitiesposition: # ["Twitch kick güzelken whatsapp çalışmıyor"] ["Twitch: 0 kick : 7 whatsapp :20"]["güzelken=12"]
                    #print("--------------------")

                    entitylen = len(sentence[fiilimsiposition:].split()[0]) # güzelken = 12 güzelken len = 8
                    subsentence = sentence[:fiilimsiposition+entitylen]   # sentence[:20] whatsapp dan öncesi
                    tempb = subsentence
                    for words in entity: #sentence is Twitch kick güzelken beforeentitiesposition is [twitch pozisyonu kick pozisyonu]
                        if subsentence.find(words) == positions and len(beforeentitiesposition)!=1 : #  0 ve 7 eğer 20 eğer eşit değilse  0 7 20 yani 20 alıcak
                            tempb=tempb.replace(words,"") # entity list contains Twitch kick turkiye.
                    sentenceresults.append(tempb)

                for positions in afterentitiesposition:
                    #print("////////////////////")

                    entitylen = len(sentence[fiilimsiposition:].split()[0])
                    subsentence = sentence[fiilimsiposition+entitylen:]
                    tempa = subsentence
                    for words in entity: # küçük bir hata var
                        if subsentence.find(words) == positions : # with last part after the and  if the entity  is less than the position
                            tempa = tempa.replace(words,"")
                    sentenceresults.append(tempa)
        else:
            sentenceresults.append(sentence)
    else:
        sentenceresults.append(sentence)
       
    temp_new_sentence = []
    for sentence in sentenceresults:
        for i in range(len(copy_entity)):
            new_sentence = re.sub("".join(entity[i]), "".join(copy_entity[i]), sentence)
        temp_new_sentence.append(new_sentence)

    return temp_new_sentence

def flatten_out_nested_list(input_list):
    if input_list is None:
        return None
    if not isinstance(input_list, (list, tuple)):
        return None
    flattened_list = []
    for entry in input_list:
        entry_list = None
        if not isinstance(entry, list):
            try:
                entry_list = ast.literal_eval(entry)
            except:
                pass
        if not entry_list:
            entry_list = entry
        if isinstance(entry_list, list):
            flattened_entry = flatten_out_nested_list(entry_list)
            if flattened_entry:
                flattened_list.extend(flattened_entry)
        else:
            flattened_list.append(entry)
    return flattened_list

def fix_nonentity_sentences(sentence_result):
    temp_sentence = ""
    new_sentence_result = []
    for i in range(len(sentence_result)):
        entity = ab_model(str(sentence_result[i]))
        if(len(entity) > 0):
            new_sentence_result.append(temp_sentence)
            temp_sentence = "".join(sentence_result[i])
        else:
            temp_sentence = temp_sentence + " " + "".join(sentence_result[i])
    new_sentence_result.append(temp_sentence)
    if("" in new_sentence_result):
        new_sentence_result.remove("")

    return new_sentence_result


def Dependency_Parser(entities:list, text:str):
    result = get_split_sentences_with_conjunction(entities, text)
    sentence_results = []
    result = fix_nonentity_sentences(result)
    print("baglac result")
    print(result)

    for sentence in result:
        entity_list = ab_model(sentence)
        sentence_results.append(splitting(entity_list,sentence))
    return sentence_results

# MODELI YUKLEME
# model path-to-save isimli klasör içine koyulmalı
# Load tokenizer
path = "path-to-save"
bert_tokenizer = BertTokenizer.from_pretrained(path +'/Tokenizer')

# Load model
bert_model = TFBertForSequenceClassification.from_pretrained(path +'/Model')

# KULLANICI GIRISI ILE TAHMIN FONKSIYONU

label = {
	0: 'Olumsuz',
	1: 'Olumlu',
    2: 'Nötr'
}

def Get_sentiment(Review, Tokenizer=bert_tokenizer, Model=bert_model):
	# Convert Review to a list if it's not already a list
	if not isinstance(Review, list):
		Review = [Review]

	Input_ids, Token_type_ids, Attention_mask = Tokenizer.batch_encode_plus(Review,
																			padding=True,
																			truncation=True,
																			max_length=1024,
																			return_tensors='tf').values()
	prediction = Model.predict([Input_ids, Token_type_ids, Attention_mask])

	# Use argmax along the appropriate axis to get the predicted labels
	pred_labels = tf.argmax(prediction.logits, axis = 1)

	# Convert the TensorFlow tensor to a NumPy array and then to a list to get the predicted sentiment labels
	pred_labels = [label[i] for i in pred_labels.numpy().tolist()]
	return pred_labels

def my_main(input_sentence):
    sentence = input_sentence
    entities = []
    entities = ab_model(sentence)
    print(entities)
    old_sentence_result = Dependency_Parser(entities, sentence)
    sentence_result = flatten_out_nested_list(old_sentence_result)
    print("fiilimsi result")
    print(sentence_result)
    
    with open("data.json", "w", encoding='utf-8') as file:
        data = {
            "sentence": sentence
        }
        entity_list = []
        sentiment_list = []
        for sentence in sentence_result:
            # Create a dictionary for each sentence
            if sentence == []:
                continue
            else:
                entities_list = ab_model("".join(sentence))
                label_list = ab_model_with_label("".join(sentence))
                if len(sentence) == 1:
                    print(sentence)
                    print("Entity:")
                    print(ab_model("".join(sentence)))
                    if(len(ab_model("".join(sentence))) != 0):
                        for i in range(len(entities_list)):
                            temp_entity = "".join(entities_list[i]) + " -> " + "".join(label_list[i])
                            temp_sentiment = "".join(entities_list[i]) + " -> " + "".join(Get_sentiment(sentence))
                            entity_list.append(temp_entity)
                            sentiment_list.append(temp_sentiment)
                else:
                    print(sentence)
                    print("Entity:")
                    print(ab_model("".join(sentence)))
                    if(len(ab_model("".join(sentence))) != 0):
                        for i in range(len(entities_list)):
                            temp_entity = "".join(entities_list[i]) + " -> " + "".join(label_list[i])
                            temp_sentiment = "".join(entities_list[i]) + " -> " + "".join(Get_sentiment(sentence))
                            entity_list.append(temp_entity)
                            sentiment_list.append(temp_sentiment)

        # Write the JSON data to the file
        data["entities"] = entity_list
        data["sentiment"] = sentiment_list

        json_data = json.dumps(data, indent=4, ensure_ascii=False)
        file.write(json_data + "\n")

    print("TAMAMLANDI")
    return data

input_sentence = "vodafone müşteri hizmetleri iyi ama çekim gücü kötü."
my_main(input_sentence)

"""
# JSON DOSYASINI GOOGLE DRIVE UZERINE YUKLEME
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
import os

# Google API erişimi için gerekli kapsam (scope)
SCOPES = ['https://www.googleapis.com/auth/drive.file']

# credentials.json dosyasının bilgisayarınızdaki tam yolu
creds_path = "C:/Users/mervegulukumm/Desktop/credentials.json"


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
# output.json
# Yüklenecek dosyanın meta verileri
file_metadata = {'name': os.path.basename(file_path)}

# Dosyayı Google Drive'a yükleme
media = MediaFileUpload(file_path, mimetype='text/plain')
file = service.files().create(body=file_metadata, media_body=media, fields='id').execute()

print(f"Dosya başarıyla yüklendi. Dosya ID'si: {file.get('id')}")
"""
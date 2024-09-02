# -*- coding: utf-8 -*-
"""main.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1kTAq27HQhPDWklLMY0FLM4Ctzl1aXBuC
"""

!pip install https://huggingface.co/turkish-nlp-suite/tr_core_news_trf/resolve/main/tr_core_news_trf-1.0-py3-none-any.whl

import spacy
import json

import transformers
from transformers import BertTokenizer, TFBertForSequenceClassification

nlp = spacy.load('tr_core_news_trf')

sentence = "Vodafone Kick kullanırken Youtube çok kasıyordu ama Turkcell kullanırken sıkıntı yoktu."
entities = []

def ab_model(test_texts):
  load_path = 'spacy_trained_model' # modelin linkini koy
  nlp = spacy.load(load_path)
  result = []
  for text in test_texts:
    doc = nlp(text)
    print(f"Text: {text}")
    print("Entities:")
    for ent in doc.ents:
        print(f"  {ent.text}: {ent.label_}")
        result.append(ent.text)

  return result

entities = ab_model(sentence)
entities

def get_split_sentences_with_conjunction(entities1, text):

    doc = nlp(text) # metnin tokenize haline getirilmesi
    split_sentences = [] # çıkarılan cümlelerin tutulacağı dizinin başlatılması
    add_old_entity = False # ilk durumda eski entity eklenmesi bool değeri False verilir.

    for token in doc: # metin içindeki her bir kelime için:
      sentence = [] # elde edilen metnin tutulacağ dizinin başlatılması
      entities = [token for token in doc if token.text in entities1] # entity lerin metin içinden tanınması
      if token.is_sent_end and token in entities: # token en sonda ise ve bir entity ise:
        entity = doc[token.i] # entity metin içinden alınır ve bir değişkende tutulur
        text = doc[:token.i].text.strip() # sondaki entityden önceki bütün metin bir değişkende tutulur
        sentence.append(entity.text + " " + text) # başa sondaki entity gelecek şekilde bütün metin entity nin sonunda eklenir
        return sentence # son metnin tutulduğu dizi döndürülür

    for sent in doc.sents: # metnin her bir cümlesi için:
        conjunctions = [token for token in sent if token.pos_ in ['CCONJ', 'SCONJ'] or token.text == "\0"] # bağlaçların metin içinden tanınması
        entities = [token for token in sent if token.text in entities1] # entity lerin metin içinden tanınması
        k = 0 # ilk bağlaç için bir sayaç oluşturulması
        if conjunctions: # metinde bağlaç var ise:
          for conjunction in conjunctions: # her bir bağlaç için:
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
          split_sentences.append(sent[conjunction.i + 1:].text.strip()) # son baglactan sonra kalan metin alınır

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

def splitting(entity : list, sentence: str):

    sentenceresults = list()

    if sentence.endswith("."):
        sentence = sentence[0:-1]

    numberofentities=howmanyentity(sentence=sentence,entity=entity) #change this function to handle empty spaces in the end
    print(f"numberofentities of this function is : {numberofentities}")
    if ( numberofentities == 1): # if there is only 1 entity
        sentenceresults.append(sentence)

    elif(numberofentities >1 ):
        numberoffiilimsi = len(find_fiilimsi(sentence=sentence))
        print(f"Number of fiilimsi is :{numberoffiilimsi}")

        if(numberoffiilimsi==1) :
            print("Hamza")
            entitiypositions = []
            for i in entity:
                entitiypositions.append(sentence.index(i))

            fiilimsiposition = sentence.find(find_fiilimsi(sentence=sentence)[0])
            print("Emin")
            whereisfiilimsi = all(index < fiilimsiposition for index in entitiypositions)
            if whereisfiilimsi:


                for i in entitiypositions:
                    entitylen = len(sentence[i:].split()[0])
                    print(f" Empty len is {entitylen}")
                    print(f"entitiyposition: {entitylen}, fiilimsipositions: {fiilimsiposition}")
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
                    print("--------------------")

                    entitylen = len(sentence[fiilimsiposition:].split()[0]) # güzelken = 12 güzelken len = 8
                    subsentence = sentence[:fiilimsiposition+entitylen]   # sentence[:20] whatsapp dan öncesi
                    tempb = subsentence
                    for words in entity: #sentence is Twitch kick güzelken beforeentitiesposition is [twitch pozisyonu kick pozisyonu]
                        if subsentence.find(words) == positions and len(beforeentitiesposition)!=1 : #  0 ve 7 eğer 20 eğer eşit değilse  0 7 20 yani 20 alıcak
                            tempb=tempb.replace(words,"") # entity list contains Twitch kick turkiye.
                    sentenceresults.append(tempb)



                for positions in afterentitiesposition:
                    print("////////////////////")

                    entitylen = len(sentence[fiilimsiposition:].split()[0])
                    subsentence = sentence[fiilimsiposition+entitylen:]
                    tempa = subsentence
                    for words in entity: # küçük bir hata var
                        if subsentence.find(words) == positions : # with last part after the and  if the entity  is less than the position
                            tempa = tempa.replace(words,"")
                    sentenceresults.append(tempa)



    return sentenceresults

def find_entities(entities,sentence):
    entit = []
    for word in sentence.split(" "):
        if word in entities:
            entit.append(word)
    return entit

sentence1 ="Vodafone RedTarife güzelken çok pahalılar"
entity = ["Vodafone","Turkcell","RedTarife"]
find_entities(entity,sentence1)

def Dependency_Parser(entities:list, text:str):
    result = get_split_sentences_with_conjunction(entities, text)
    sentence_results = []

    for sentence in result:
        entity_list = find_entities(entities,sentence)
        sentence_results.append(splitting(entity_list,sentence))
    return sentence_results

sentence_result = []
sentence_result = Dependency_Parser(entities, sentence)

# MODELI YUKLEME
# model path-to-save isimli klasör içine koyulmalı
# Load tokenizer
bert_tokenizer = BertTokenizer.from_pretrained(path +'/Tokenizer')

# Load model
bert_model = TFBertForSequenceClassification.from_pretrained(path +'/Model')

# KULLANICI GIRISI ILE TAHMIN FONKSIYONU

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

with open("data.json", "a+") as file:
    for sentence in sentence_result:
        # Create a dictionary for each sentence
        data = {
            "sentence": sentence,
            "entities": ab_model(sentence),
            "sentiment": Get_sentiment(sentence)
        }

        # Convert the dictionary to a JSON string
        json_data = json.dumps(data, indent=4)

        # Write the JSON data to the file
        file.write(json_data + "\n")  # Add a newline to separate JSON objects

        # Print the sentiment (optional)


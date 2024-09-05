import spacy
import json
import re
import tensorflow as tf
from transformers import BertTokenizer, TFBertForSequenceClassification


nlp = spacy.load('tr_core_news_trf')

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
    notallowed = ["vodafone","Vodafone","VODAFONE","Turkcell","Türkcell","türkcell","turkcell","Türktelekom"]
    words = sentence.split(" ")
    for token in doc:
        # Check if the token ends with any of the fiilimsi suffixes
        if any(token.text.endswith(suffix) for suffix in fiilimsi_suffixes):
            # Check dependency and POS tags to filter fiilimsi
            if (token.dep_ in ["advcl", "ccomp", "amod", "acl"] or (token.pos_ in ['VERB', 'AUX'] and token.dep_ not in ["ROOT"])) and (token.text not in notallowed) and (token.text != words[len(words)-1]):
                fiilimsis.append(token.text)

    return fiilimsis

def clean_text_for_dependency_parsing(sentence):
    import string
    keep_punctuation = {'.', '?'}

    cleaned_sentence = ''.join(char for char in sentence if char not in string.punctuation or char in keep_punctuation)
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
    # copy_entity = entity.copy()
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
        if(numberoffiilimsi == 0): # 
          poss = []
          for tempen in entity:
            poss.append(sentence.index(tempen))

# Iterate over the positions of entities
          for i in range(len(poss)-1):
            position1 = poss[i]
            position2 = poss[i+1]
    
    # Calculate the length of the first entity
            entitylen1 = len(sentence[position1:].split()[0])
    
    # Split the sentence if entities are more than 15 characters apart
            if position2 - position1 > 15:
                sentence1 = sentence[position1:position2].strip()  # Take the part between the entities
                sentenceresults.append(sentence1)
    
    # Always take the second part of the sentence after the last entity
            if i == len(poss) - 2:
                sentence2 = sentence[position2:].strip()
                sentenceresults.append(sentence2)
            
            
        if(numberoffiilimsi==1) :
           
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

    
    return sentenceresults

#print(splitting(["vodafone","whatsapp","instagram","facebook"],"vodafone whatsapp, instagram, facebook, twitter  sınırsız olduğu söylendi bana. fakat youtube.com ve sosyal medyaları kullandığım zaman 30 GB olan internetimden düşüş sağlanıyor."))# spacy
#print(splitting(["vodafone","whatsapp","türkcell"], "vodafone çok iyi çok muhteşem whatsapp çok kötü türkcell allah belanı versin")) # doğru 
#print(splitting(["vodafone","whatsapp"], "ahmet çok iyi mehmet çok kötü")) # doğru
#print(splitting([], "ahmet çok iyi mehmet çok kötü")) # doğru

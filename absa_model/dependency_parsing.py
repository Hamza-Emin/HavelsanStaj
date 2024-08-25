import spacy

nlp = spacy.load("tr_core_news_md")
def split_sentence(doc):
    sentences = []
    current_sentence = []
    
    for token in doc:
        current_sentence.append(token.text)
        
    
        if token.dep_ in ['cc', 'punct'] and token.text.lower() not in ['ve', 'ancak']:
            sentences.append(" ".join(current_sentence).strip())
            current_sentence = []
    

    if current_sentence:
        sentences.append(" ".join(current_sentence).strip())
    
    return sentences

text = "Vodafone berbat , türkcell çok iyi. Türk telekom mükkemmel"
doc = nlp(text)
split_sentences = split_sentence(doc)
for sentence in split_sentences:
    print(sentence)

import pandas as pd
import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import joblib
import numpy as np


mlb = joblib.load('mlb.pkl')


model = DistilBertForSequenceClassification.from_pretrained('aspects_model')
tokenizer = DistilBertTokenizer.from_pretrained('aspects_model')


def predict_tags(model, tokenizer, text, threshold=0.5):
    encoding = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=128,
        return_token_type_ids=False,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt',
    )

    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)

    model.eval()
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        probs = torch.sigmoid(logits).squeeze().cpu().numpy()
        predicted_tags = [mlb.classes_[i] for i, p in enumerate(probs) if p > threshold]

    return predicted_tags

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model = model.to(device)

new_sentence = "Vodafone interneti kötü, mkşteri hizmetleri çalışmıyor, fiyat çok yüksek."
predicted_tags = predict_tags(model, tokenizer, new_sentence)
print(predicted_tags)

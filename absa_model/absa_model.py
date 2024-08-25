import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, AdamW
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import numpy as np
import joblib


df = pd.read_csv('output_encoded.csv', sep=';')
df = df.drop(columns=['Aspects'],axis=1)

mlb = joblib.load('mlb.pkl')


class AspectDataset(Dataset):
    def __init__(self, comments, labels, tokenizer, max_len):
        self.comments = comments
  
        self.labels = torch.tensor(labels, dtype=torch.float32)
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.comments)

    def __getitem__(self, index):
        comment = str(self.comments[index])
        label = self.labels[index]

        encoding = self.tokenizer.encode_plus(
            comment,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': label
        }


tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=len(mlb.classes_))


train_texts, val_texts, train_labels, val_labels = train_test_split(df['Clean'], df[mlb.classes_].values, test_size=0.2, random_state=42)

train_dataset = AspectDataset(train_texts.tolist(), train_labels, tokenizer, max_len=128)
val_dataset = AspectDataset(val_texts.tolist(), val_labels, tokenizer, max_len=128)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16)


optimizer = AdamW(model.parameters(), lr=5e-5)


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model = model.to(device)

epochs = 5
for epoch in range(epochs):
    model.train()
    for batch in train_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()

    print(f'Epoch {epoch+1}/{epochs}, Loss: {loss.item()}')

# Save the model
model.save_pretrained('aspects_model3')
tokenizer.save_pretrained('aspects_model3')


def evaluate_model(model, data_loader):
    model.eval()
    predictions, true_labels = [], []
    
    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits

            probs = torch.sigmoid(logits).cpu().numpy()
            predictions.extend(probs)
            true_labels.extend(labels.cpu().numpy())

    return np.array(predictions), np.array(true_labels)


val_predictions, val_true_labels = evaluate_model(model, val_loader)


threshold = 0.5
val_predictions = (val_predictions > threshold).astype(int)


accuracy = accuracy_score(val_true_labels, val_predictions)
precision, recall, f1, _ = precision_recall_fscore_support(val_true_labels, val_predictions, average='weighted')

print(f'Validation Accuracy: {accuracy}')
print(f'Validation Precision: {precision}')
print(f'Validation Recall: {recall}')
print(f'Validation F1 Score: {f1}')


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


new_sentence = "Vodafone internet kötü , müşteri hizmetleri çalışmıyor , fiyat çok yüksek."
predicted_tags = predict_tags(model, tokenizer, new_sentence)
print(predicted_tags)

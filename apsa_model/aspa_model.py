import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split

# Load your CSV
df = pd.read_csv('output.csv',on_bad_lines='skip',sep=';')
df = df.drop(columns='Unnamed: 0',axis=1)
print("-----------------------------")
print(df.columns)

# Prepare labels for multi-label classification
mlb = MultiLabelBinarizer()
df['Aspect_encoded'] = mlb.fit_transform(df['Aspects']).tolist()

# Define Dataset class
class AspectDataset(Dataset):
    def __init__(self, comments, labels, tokenizer, max_len):
        self.comments = comments
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.comments)

    def __getitem__(self, index):
        comment = str(self.comments[index])
        label = torch.tensor(self.labels[index], dtype=torch.float)

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

# Initialize tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=len(mlb.classes_))

# Prepare data for training
train_texts, val_texts, train_labels, val_labels = train_test_split(df['Explanation'], df['Aspect_encoded'], test_size=0.2)
train_dataset = AspectDataset(train_texts.tolist(), train_labels.tolist(), tokenizer, max_len=128)
val_dataset = AspectDataset(val_texts.tolist(), val_labels.tolist(), tokenizer, max_len=128)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16)

# Define optimizer
from torch.optim import AdamW

optimizer = AdamW(model.parameters(), lr=5e-5)


# Training loop
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model = model.to(device)

epochs = 3
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
model.save_pretrained('aspect_model')
tokenizer.save_pretrained('aspect_model')

# Predicting on new data
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

# Example prediction
new_sentence = "Vodafone's pricing is terrible."
predicted_tags = predict_tags(model, tokenizer, new_sentence)
print(predicted_tags)

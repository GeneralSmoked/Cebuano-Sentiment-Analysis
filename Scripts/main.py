import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification
import torch
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

#load dataset
df = pd.read_csv(r'C:\Users\Rosh\Documents\GitHub\Cebuano Sentiment Analysis\Data\CebuanoDataset.csv')
#load tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

train_texts, test_texts, train_labels, test_labels = train_test_split(df['text'],df['label'], test_size = 0.2, random_state=42)

train_tokens = tokenizer(train_texts.tolist(), padding = True, truncation = True, return_tensors= 'pt')
test_tokens = tokenizer(test_texts.tolist(), padding = True, truncation = True, return_tensors= 'pt')

train_labels = torch.tensor(train_labels.tolist())
test_labels = torch.tensor(test_labels.tolist())

train_dataset = TensorDataset(train_tokens['input_ids'], train_tokens['attention_mask'], train_labels)
test_dataset = TensorDataset(test_tokens['input_ids'], test_tokens['attention_mask'], test_labels)

#dataloaders
train_dataloader = DataLoader(train_dataset, batch_size = 16, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size = 16)

#Load Model
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
optimizer = AdamW(model.parameters(), lr = 1e-5)

#Train Model
model.train()
for epoch in range(3):
    total_loss = 0
    for batch in train_dataloader:
        input_ids, attention_mask, labels = batch
        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask = attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        print(f'Epoch {epoch+1}, Loss: {total_loss/len(train_dataloader)}')

model.save_pretrained('Cebuano_sentiment_analysis_model')
tokenizer.save_pretrained('Cebuano_sentiment_analysis_tokenizer')

#Evaluate Model
model.eval()
all_preds = []
all_true = []

with torch.no_grad():
    for batch in test_dataloader:
        input_ids, attention_mask, labels = batch
        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        preds = torch.argmax(logits, dim=1)
        all_preds.extend(preds.tolist())
        all_true.extend(labels.tolist())

print("\n --- Evaluation Results ---")
print("Accuracy: ", accuracy_score(all_true, all_preds))
print("Confusion Matrix:\n", confusion_matrix(all_true, all_preds))
print("Classification Report:\n", classification_report(all_true, all_preds))
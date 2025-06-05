from transformers import BertTokenizer, BertForSequenceClassification
import torch

tokenizer = BertTokenizer.from_pretrained('Cebuano_sentiment_analysis_tokenizer')
model = BertForSequenceClassification.from_pretrained('Cebuano_sentiment_analysis_model')

model.eval()

phrase = ["lipaya nako rung adlawa"]

tokens = tokenizer(phrase, padding=True, truncation = True, return_tensors = 'pt')

with torch.no_grad():
    output = model(**tokens)
    logits = output.logits

predicted_class = torch.argmax(logits, dim=1).item()

if predicted_class == 1:
    print(f'The phrase: "{phrase}" is Positive')
else:
    print(f'The phrase: "{phrase}" is Negative')


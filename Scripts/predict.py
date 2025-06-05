from transformers import BertTokenizer, BertForSequenceClassification
from tqdm import tqdm
import torch
import pandas as pd

df_test = pd.read_csv(r'C:\Users\Rosh\Documents\GitHub\Cebuano Sentiment Analysis\Data\CebuanoTestSet.csv')

tokenizer = BertTokenizer.from_pretrained('Cebuano_sentiment_analysis_tokenizer')
model = BertForSequenceClassification.from_pretrained("Cebuano_sentiment_analysis_model")

model.eval()

def predict_sentiment(text,tokenizer,model):
    tokens = tokenizer (text, padding=True, truncation=True, return_tensors='pt')

    with torch.no_grad():
        output = model(**tokens)
        logits = output.logits
    
    predicted_class = torch.argmax(logits, dim=1).item()
    if predicted_class == 1:
        print(f'The phrase: "{text}" is Positive')
    else:
        print(f'The phrase: "{text}" is Negative')
        return predicted_class
    
tqdm.pandas()
df_test['predicted_sentiment_text'] = df_test.progress_apply(lambda x: predict_sentiment(x['Text'], tokenizer, model),axis =1)
df_test['predicted'] = df_test['Text'].apply(lambda x: predict_sentiment(x,tokenizer,model))
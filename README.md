Cebuano Sentiment Analysis with BERT
This project is my personal entry into learning more about Natural Language Processing (NLP), specifically focused on Cebuano text sentiment analysis. It uses a pretrained BERT base model for both training and tokenization.

📊 Dataset
The dataset used in this project is a manually annotated corpus, created and labeled entirely by hand. It also includes a small portion of synthetically generated data to help balance and augment the dataset. The sentiment labels are binary — either positive or negative — with no neutral category included.

⚙️ Setup
To use this project:

Create a virtual environment (recommended):
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

Install dependencies:
pip install -r requirements.txt
(Optional) If using CUDA for GPU acceleration, follow the installation steps in installs.txt.

🧪 Scripts
You can explore and modify the following scripts based on your needs:

main.py — For training the model
inference.py — For running inference on new text
cebuano_sentiment_analysis_model/ — Trained model files
cebuano_sentiment_analysis_tokenizer/ — Tokenizer files

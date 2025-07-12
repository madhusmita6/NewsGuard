import os
import re
import string
import joblib
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

nltk.download('punkt')
nltk.download('stopwords')

# Setup paths
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model_path = os.path.join(base_dir, "models", "model.pkl")
vect_path = os.path.join(base_dir, "models", "vectorizer.pkl")

# Load model and vectorizer
model = joblib.load(model_path)
vectorizer = joblib.load(vect_path)

STOPWORDS = set(stopwords.words('english'))

def clean_text(text):
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    tokens = word_tokenize(text)
    filtered_tokens = [word for word in tokens if word not in STOPWORDS]
    return " ".join(filtered_tokens)

def predict_news(text):
    clean = clean_text(text)
    vect = vectorizer.transform([clean])
    prediction = model.predict(vect)[0]
    label = "Fake News ❌" if prediction == 0 else "Real News ✅"
    return label

# Demo run
if __name__ == "__main__":
    sample_text = input("📰 Enter a news article text:\n")
    result = predict_news(sample_text)
    print("\n🔎 Prediction:", result)

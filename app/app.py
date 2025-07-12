import streamlit as st
import os
import joblib
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Ensure NLTK assets are available
nltk.download('punkt')
nltk.download('stopwords')
STOPWORDS = set(stopwords.words('english'))

# Load model and vectorizer
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model = joblib.load(os.path.join(base_dir, "models", "model.pkl"))
vectorizer = joblib.load(os.path.join(base_dir, "models", "vectorizer.pkl"))

# Clean user input text
def clean_text(text):
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    tokens = word_tokenize(text)
    filtered = [word for word in tokens if word not in STOPWORDS]
    return " ".join(filtered)

# App UI
st.set_page_config(page_title="NewsGuard", page_icon="🛡️")
st.title("🛡️ NewsGuard – Fake News Detection")
st.markdown("Enter a news article or headline below to check if it's **real** or **fake**.")

# Input box
user_input = st.text_area("📰 Paste news content here:", height=200)

# Predict button
if st.button("🔍 Predict"):
    if user_input.strip() == "":
        st.warning("Please enter some news content.")
    else:
        clean = clean_text(user_input)
        vect = vectorizer.transform([clean])
        pred = model.predict(vect)[0]
        prob = model.predict_proba(vect)[0]

        if pred == 0:
            st.error("❌ This looks like **Fake News**.")
        else:
            st.success("✅ This appears to be **Real News**.")

        st.markdown(f"**Confidence:** {prob[pred]*100:.2f}%")

st.markdown("---")
st.markdown("Built with ❤️ using Streamlit and MLflow")

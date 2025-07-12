import pandas as pd
import string
import re
import os
from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Download stopwords/tokenizer once
import nltk
nltk.download('punkt')
nltk.download('stopwords')

STOPWORDS = set(stopwords.words('english'))

def clean_text(text):
    """Lowercase, remove punctuation, digits, and stopwords."""
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    tokens = word_tokenize(text)
    filtered_tokens = [word for word in tokens if word not in STOPWORDS]
    return " ".join(filtered_tokens)

def load_and_process_data(fake_path, real_path, save_to=None):
    """Loads, cleans, splits, and optionally saves the dataset."""
    df_fake = pd.read_csv(fake_path)
    df_real = pd.read_csv(real_path)

    df_fake['label'] = 0
    df_real['label'] = 1

    df = pd.concat([df_fake, df_real], ignore_index=True)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    df['text'] = df['text'].astype(str).apply(clean_text)

    train, test = train_test_split(df, test_size=0.2, random_state=42)

    if save_to:
        os.makedirs(save_to, exist_ok=True)
        train.to_csv(os.path.join(save_to, 'train.csv'), index=False)
        test.to_csv(os.path.join(save_to, 'test.csv'), index=False)

    return train, test


if __name__ == "__main__":
    # Use absolute paths based on project root
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    fake_path = os.path.join(base_dir, "data", "raw", "Fake.csv")
    real_path = os.path.join(base_dir, "data", "raw", "True.csv")
    save_to = os.path.join(base_dir, "data", "processed")

    load_and_process_data(fake_path, real_path, save_to)
    print("✅ Data preprocessing complete. Cleaned files saved.")

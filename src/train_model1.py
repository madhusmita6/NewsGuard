import pandas as pd
import os
import mlflow
import mlflow.sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from joblib import dump
import warnings

warnings.filterwarnings("ignore")

# Paths
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
train_path = os.path.join(base_dir, "data", "processed", "train.csv")
model_dir = os.path.join(base_dir, "models")
os.makedirs(model_dir, exist_ok=True)

# Load and clean data
df = pd.read_csv(train_path)
df = df.dropna(subset=['text', 'label'])  # Remove NaNs
X = df['text'].astype(str)                # Ensure all entries are strings
y = df['label']


# TF-IDF Vectorization
vectorizer = TfidfVectorizer(max_features=5000)
X_vec = vectorizer.fit_transform(X)

# Define model
model = LogisticRegression()

# MLflow logging
mlflow.set_experiment("NewsGuard Fake News Detection")

with mlflow.start_run():
    mlflow.log_param("vectorizer", "TfidfVectorizer")
    mlflow.log_param("model_type", "LogisticRegression")

    model.fit(X_vec, y)

    # Predictions for evaluation
    y_pred = model.predict(X_vec)
    acc = accuracy_score(y, y_pred)
    f1 = f1_score(y, y_pred)

    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("f1_score", f1)

    # Save model and vectorizer to disk
    model_path = os.path.join(model_dir, "model.pkl")
    vect_path = os.path.join(model_dir, "vectorizer.pkl")

    dump(model, model_path)
    dump(vectorizer, vect_path)

    # Log model with MLflow
    mlflow.sklearn.log_model(model, "model")

print("✅ Model training complete. Files saved and run logged to MLflow.")

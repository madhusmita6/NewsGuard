import os
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from joblib import dump

# === Load and preprocess data ===
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
train_path = os.path.join(base_dir, "data", "processed", "train.csv")
model_dir = os.path.join(base_dir, "models")
os.makedirs(model_dir, exist_ok=True)

df = pd.read_csv(train_path)
df = df.dropna(subset=['text', 'label'])
X = df['text'].astype(str)
y = df['label']

# === TF-IDF Vectorization ===
vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
X_vec = vectorizer.fit_transform(X)

# === MLflow Experiment ===
mlflow.set_experiment("NewsGuard Fake News Detection")

with mlflow.start_run():
    # ✅ Log parameters BEFORE training
    mlflow.log_param("model_type", "LogisticRegression")
    mlflow.log_param("vectorizer", "TfidfVectorizer")
    mlflow.log_param("max_features", 5000)
    mlflow.log_param("stop_words", "english")
    mlflow.log_param("class_weight", "balanced")

    # === Train model ===
    model = LogisticRegression(class_weight='balanced')
    model.fit(X_vec, y)

    # === Evaluate ===
    y_pred = model.predict(X_vec)
    acc = accuracy_score(y, y_pred)
    f1 = f1_score(y, y_pred)

    # ✅ Log metrics
    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("f1_score", f1)

    # ✅ Log model as artifact
    mlflow.sklearn.log_model(model, "model")

    # ✅ Save to local disk as well
    dump(model, os.path.join(model_dir, "model.pkl"))
    dump(vectorizer, os.path.join(model_dir, "vectorizer.pkl"))

print("✅ Model training complete and logged to MLflow.")

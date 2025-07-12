import os
import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn

# Paths
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
test_path = os.path.join(base_dir, "data", "processed", "test.csv")
model_path = os.path.join(base_dir, "models", "model.pkl")
vect_path = os.path.join(base_dir, "models", "vectorizer.pkl")

# Load artifacts
model = joblib.load(model_path)
vectorizer = joblib.load(vect_path)

# Load test data
df = pd.read_csv(test_path)
df = df.dropna(subset=['text', 'label'])
X_test = df['text'].astype(str)
y_test = df['label']

# Vectorize
X_test_vec = vectorizer.transform(X_test)

# Predict
y_pred = model.predict(X_test_vec)

# Metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# Print classification report
print("🔍 Classification Report:\n")
print(classification_report(y_test, y_pred))

# Plot confusion matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Fake', 'Real'], yticklabels=['Fake', 'Real'])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.tight_layout()

# Save plot
eval_plot_path = os.path.join(base_dir, "models", "confusion_matrix.png")
plt.savefig(eval_plot_path)
plt.show()

# Log to MLflow
with mlflow.start_run(run_name="Evaluation"):
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("f1_score", f1)
    mlflow.log_artifact(eval_plot_path)

print("\n✅ Evaluation complete. Metrics logged to MLflow.")

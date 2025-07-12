# 🛡️ NewsGuard – Fake News Detection with MLOps

**NewsGuard** is an end-to-end fake news detection application built with:
- 🧠 Machine Learning for text classification
- 📊 MLflow for experiment tracking and model lifecycle management
- 🌐 Streamlit for an interactive web UI
- 🐳 Docker for containerized deployment
- ☁️ cloud deployment on Render (https://fake-news-detector-qrok.onrender.com/)

---

## 🚀 Project Structure

```bash
NewsGuard/
│
├── app/                       # Streamlit frontend
│   └── app.py
│
├── data/                      # Dataset (True.csv, Fake.csv, etc.)
│   ├── raw/
│   └── processed/
│
├── models/                    # Saved models and vectorizers
│
├── src/                       # Core scripts
│   ├── train_model.py         # ML model training script
│   ├── train_transformer.py   # RoBERTa/BERT training (optional)
│   ├── evaluation.py          # Model evaluation
│   └── mlruns/                # MLflow run logs
│
├── requirements.txt
├── Dockerfile
├── .gitignore
└── README.md

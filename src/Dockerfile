# Use the official lightweight Python image
FROM python:3.10-slim

# Set working directory inside container
WORKDIR /app
# Use a lightweight Python image
FROM python:3.10-slim

# Set working directory inside the container
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN apt-get update && apt-get install -y build-essential && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Copy everything else (including nltk_data)
COPY . .

# Set environment variable for NLTK data location
ENV NLTK_DATA=/app/nltk_data

# Expose Streamlit default port
EXPOSE 8501

# Run Streamlit app
#CMD ["streamlit", "run", "src/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
#CMD ["python", "-m", "streamlit", "run", "src/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
CMD ["python", "-m", "streamlit", "run", "app/app.py", "--server.port=8501", "--server.address=0.0.0.0"]



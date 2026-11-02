
FROM python:3.12-slim

# Dossier de travail
WORKDIR /app

# Dépendances système nécessaires
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Installer les dépendances Python
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# Copier le code et l'index FAISS
COPY RAMchat.py .
COPY retriever.py .
COPY faiss_vectoriel_bd/ ./faiss_vectoriel_bd/

# Port Streamlit
EXPOSE 8501

# Lancer l'application
ENTRYPOINT ["streamlit", "run", "RAMchat.py", "--server.address=0.0.0.0", "--server.port=8501"]

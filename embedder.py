from langchain_mistralai import MistralAIEmbeddings
from langchain_community.vectorstores import FAISS
import json, os
from dotenv import load_dotenv
load_dotenv()

MISTRAL_API_KEY=os.getenv("MISTRAL_API_KEY")

embeddings = MistralAIEmbeddings(
    model="mistral-embed-2312",
    api_key=MISTRAL_API_KEY
)

with open("royal_air_maroc_data.json", "r", encoding="utf-8") as f:
    data = json.load(f)


texts = []
metadatas = []

for item in data:

    result = item.get("result", {})

    page_url = item.get("url", "")
    page_title = result.get("page_title", "")

    chunks = result.get("chunks", [])

    for chunk in chunks:

        content = chunk.get("content", "").strip()

        if not content:
            continue

        texts.append(content)

        metadatas.append({
            "source": chunk.get("source_url", page_url),
            "page_title": page_title,
            "chunk_id": chunk.get("chunk_id"),
            "section_title": chunk.get("section_title", "")
        })

print("Nombre total de chunks :", len(texts))
print("Nombre de métadonnées :", len(metadatas))

vector_store = FAISS.from_texts(
    texts=texts,
    embedding=embeddings,
    metadatas=metadatas
)

vector_store.save_local("faiss_vectoriel_bd_2")
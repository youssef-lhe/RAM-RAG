import json

with open("royal_air_maroc_data.json", "r", encoding="utf-8") as f:
    data = json.load(f)

target_url = "https://www.royalairmaroc.com/ma-fr/flotte"

for item in data:
    if item.get("url") == target_url:

        result = item.get("result", {})

        # Afficher tous les chunks
        chunks = result.get("chunks", [])

        for chunk in chunks:
            print(f"\n--- Chunk {chunk.get('chunk_id')} ---")
            print(chunk.get("content", ""))

import requests
import json
import re

from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse

SITEMAP_URL = "https://www.royalairmaroc.com/ma-fr/site-map"
DOMAIN = "www.royalairmaroc.com"

headers = {
    "User-Agent": "Mozilla/5.0"
}

response = requests.get(
    SITEMAP_URL,
    headers=headers,
    timeout=30
)

response.raise_for_status()

soup = BeautifulSoup(response.text, "html.parser")

urls = {}

# Expressions pour identifier les destinations
destination_pattern = re.compile(
    r"^(vols?\s+vers|flights?\s+(to|from)|voli\s+da)\b",
    re.IGNORECASE
)

for a in soup.find_all("a", href=True):

    href = a["href"].strip()
    text = a.get_text(" ", strip=True)

    url = urljoin(SITEMAP_URL, href)
    parsed = urlparse(url)

    # Liens internes uniquement
    if parsed.netloc != DOMAIN:
        continue

    # Exclure les destinations par leur chemin
    if "/destinations/" in parsed.path.lower():
        continue

    # Exclure les liens de destinations par leur intitulé
    if destination_pattern.search(text):
        continue

    # Exclure les liens vides ou les ancres
    if not parsed.path or parsed.path == "/":
        continue

    clean_url = parsed._replace(
        query="",
        fragment=""
    ).geturl().rstrip("/")

    urls[clean_url] = text

# Export JSON
results = [
    {
        "title": title,
        "url": url,
        "path": urlparse(url).path
    }
    for url, title in urls.items()
]

with open("ram_site_paths.json", "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=4)

print(f"Total des chemins récupérés : {len(results)}")

for item in results:
    print(item["path"])

from scrapegraphai.graphs import SmartScraperGraph
import json, os, time
from dotenv import load_dotenv
load_dotenv()

with open("ram_site_paths.json", "r", encoding="utf-8") as f:
    data = json.load(f)

all_urls=[]
for item in data:
    all_urls.append(item["url"])


MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")

graph_config = {
    "llm": {
        
        "model": "mistralai/mistral-large-4",
        "temperature": 0,
	"api_key": MISTRAL_API_KEY,
	"model_tokens": 1000000
	
	
  
    },
    "verbose": True,
    "headless": True,
}

prompt = """
Tu es un expert en Web Scraping, Web Content Extraction et préparation
de données pour les systèmes Retrieval-Augmented Generation (RAG).

Ta mission est d'analyser la page web fournie, de nettoyer son contenu
et d'extraire les informations pertinentes sous forme de chunks
sémantiquement cohérents pour alimenter une base de connaissances
destinée à un assistant conversationnel Royal Air Maroc.

### 1. NETTOYAGE DU CONTENU WEB

Supprime tous les éléments techniques et non pertinents, notamment :
- Le code HTML, les balises et les attributs.
- Le CSS, les styles et les classes.
- Le JavaScript et les scripts.
- Les menus de navigation et sous-menus.
- Les en-têtes et pieds de page répétitifs.
- Les bannières publicitaires et pop-ups.
- Les bandeaux de cookies.
- Les boutons, icônes et éléments purement graphiques.
- Les liens de navigation sans valeur informationnelle.
- Les contenus dupliqués et les éléments techniques.
- Supprime les retours à la ligne inutiles (\n, \r, \r\n).
- Supprime les tabulations et les espaces multiples.
- Supprime les lignes vides répétitives.
- Élimine les espaces au début et à la fin des phrases.
- Corrige les séparations artificielles de mots causées par le formatage HTML.
- Supprime les caractères invisibles et les caractères parasites.
- Évite les mots ou phrases fragmentés sur plusieurs lignes sans raison.
- Normalise les espaces entre les mots et les phrases.

Ne conserve que le contenu textuel ayant une réelle valeur
informationnelle pour un voyageur.

### 2. EXTRACTION DU CONTENU PERTINENT

Extrais fidèlement les informations présentes dans la page, notamment :
- Les paragraphes informatifs.
- Les titres et sous-titres.
- Les listes à puces et numérotées.
- Les tableaux contenant des informations utiles.
- Les FAQ et leurs réponses.
- Les procédures et instructions.
- Les conditions, restrictions et exceptions.
- Les tarifs, poids, dimensions, délais et autres valeurs numériques.
- Les informations relatives aux services et aux conditions de voyage.

Conserve les relations entre les titres et leurs paragraphes.
Ne supprime pas une information simplement parce qu'elle est répétée
si elle apporte un contexte ou une condition différente.

### 3. CRÉATION DES CHUNKS

Découpe le contenu nettoyé en chunks adaptés à la recherche
sémantique et à la récupération contextuelle dans un système RAG.

Respecte les règles suivantes :
- Chaque chunk doit traiter un sujet ou une information cohérente.
- Privilégie les frontières naturelles : paragraphes, sections,
  procédures, règles et FAQ.
- Conserve les titres et sous-titres associés au contenu.
- Ne sépare jamais une règle de ses conditions, exceptions
  ou restrictions associées.
- Évite de mélanger plusieurs sujets indépendants dans un même chunk.
- Chaque chunk doit être compréhensible indépendamment autant que possible.
- Évite les chunks trop courts ou excessivement longs.
- Ne coupe pas une phrase ou une information au milieu.

### 4. FIDÉLITÉ À LA SOURCE

Règles impératives :
- Ne résume pas le contenu.
- Ne paraphrase pas et ne reformule pas les informations.
- Ne corrige pas les erreurs du texte original.
- N'ajoute aucune information issue de tes connaissances.
- Ne déduis aucune règle qui n'est pas explicitement présente.
- Préserve exactement les valeurs numériques, tarifs, devises,
  unités, dates, conditions et restrictions.
- Ne transforme pas une recommandation en obligation.
- Ne transforme pas une possibilité en règle générale.
- N'invente aucune information manquante.

Tu peux supprimer le bruit technique et organiser le texte en chunks,
mais le contenu informationnel de chaque chunk doit rester fidèle
au texte original.

### 5. MÉTADONNÉES ET TRAÇABILITÉ

Pour chaque chunk, conserve :
- Un identifiant unique.
- Le contenu textuel du chunk.
- L'URL exacte de la page source.
- Le titre de la page, si disponible.
- Le titre de la section d'origine, si disponible.

### 6. FORMAT DE SORTIE

Retourne exclusivement un JSON valide, sans Markdown
ni commentaire supplémentaire.

Utilise exactement cette structure :

{
    "source_url": "",
    "page_title": "",
    "chunks": [
        {
            "chunk_id": 1,
            "section_title": "",
            "content": "",
            "source_url": ""
        }
    ]
}

Si aucun contenu pertinent n'est disponible, retourne une liste
de chunks vide.

OBJECTIF FINAL :

Produire des chunks propres, informatifs, cohérents sémantiquement
et fidèles aux pages officielles de Royal Air Maroc, directement
exploitables pour la génération d'embeddings et l'indexation
dans une base de données vectorielle.
"""
results = []
errors=[]

# Limites Mistral Large 4
REQUEST_INTERVAL = 2       # secondes entre les pages
MAX_RETRIES = 5
INITIAL_BACKOFF = 5        # secondes

for i, url in enumerate(all_urls, start=1):

    print(f"\nTraitement {i}/{len(all_urls)} : {url}")

    attempt = 0

    while attempt < MAX_RETRIES:
        try:
            scraper = SmartScraperGraph(
                prompt=prompt,
                source=url,
                config=graph_config
            )

            result = scraper.run()

            results.append({
                "url": url,
                "result": result
            })

            print(f"Succès : {url}")
            break

        except Exception as e:

            error_message = str(e).lower()

            # Détection des erreurs de limitation
            is_rate_limit = (
                "429" in error_message
                or "rate limit" in error_message
                or "too many requests" in error_message
            )

            if is_rate_limit:
                attempt += 1

                if attempt >= MAX_RETRIES:
                    errors.append({
                        "url": url,
                        "error": str(e)
                    })
                    print(f"Échec après {MAX_RETRIES} tentatives : {url}")
                    break

                wait_time = INITIAL_BACKOFF * (2 ** (attempt - 1))

                print(
                    f"Rate limit détecté. "
                    f"Nouvelle tentative dans {wait_time}s..."
                )

                time.sleep(wait_time)

            else:
                errors.append({
                    "url": url,
                    "error": str(e)
                })

                print(f"Erreur sur {url}: {e}")
                break

    # Respecter un intervalle entre les pages
    if i < len(all_urls):
        time.sleep(REQUEST_INTERVAL)


with open("royal_air_maroc_data.json", "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=4)
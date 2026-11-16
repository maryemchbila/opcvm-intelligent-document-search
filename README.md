# Projet_tutore_gemini

Pipeline minimal :

```text
PDF -> Gemini -> JSON structure -> validation Pydantic -> JSON local
```

Le PDF sélectionné est transmis directement à Gemini via le SDK `google-genai`
Files API. Aucune conversion OCR/texte locale, base PostgreSQL, recherche web ou
autre outil Gemini n'est utilisé. Le fichier distant temporaire est supprimé
après la génération. Les citations `source_text` et pages sont fournies par
Gemini ; sans extraction locale, elles ne peuvent pas être comparées
automatiquement au texte interne du PDF.

## Installation (Windows)

```powershell
cd Projet_tutore_gemini
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Copiez `.env.example` vers `.env`, puis renseignez `GEMINI_API_KEY`. Ne publiez
jamais `.env`, les PDF ou `output/`.

## Vérifier Gemini

```powershell
.\.venv\Scripts\python.exe scripts\legacy\extract_pdf.py --check-gemini
```

## Analyser un seul PDF

Indiquez explicitement le chemin relatif ou absolu d'un unique PDF. Les PDF
réels peuvent rester dans `Fichiers\` :

```powershell
.\.venv\Scripts\python.exe scripts\legacy\extract_pdf.py "Fichiers\NOM_DU_PDF.pdf"
```

`documents\` reste utilisable pour des PDF fictifs ou de test, mais n'est pas
obligatoire. Le programme ne parcourt jamais automatiquement `Fichiers\` ou un
autre dossier et ne lance aucun traitement par lot. Chaque exécution traite
uniquement le fichier explicitement indiqué et crée un nouveau run dans
`output\raw\`, `output\structured\` et `output\reports\`.

## Tests

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests
```

## Recherche semantique

La recherche vectorielle utilise par defaut
`sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`. Ce modele
multilingue de taille moderee couvre le francais et produit des embeddings de
384 dimensions, ce qui convient a un petit corpus et a une machine CPU avec
16 Go de RAM. Il peut etre remplace avec `EMBEDDING_MODEL` dans `.env`.

Les embeddings sont normalises puis indexes avec `faiss.IndexFlatIP` : le
produit scalaire correspond ainsi a la similarite cosinus. L'index est
reconstruit en memoire a chaque lancement.

```powershell
.\.venv\Scripts\python.exe scripts\semantic_search.py "gestion active des liquidites"
```

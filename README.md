
<<<<<<< HEAD
=======
## Structure du projet

```
.
├── app.py              # Interface Streamlit + point d'entrée (à lancer)
├── functions.py         # Logique métier : ingestion, recherche, RAG
├── model.py             # Configuration + modèles (embeddings, LLM)
├── requirements.txt      # Dépendances Python
├── run.sh                # Script de lancement (Mac/Linux)
├── run.bat               # Script de lancement (Windows)
```

## Prérequis

- Python 3.10 ou supérieur
- [Ollama](https://ollama.com/) installé localement


2. Installer les dépendances :

   ```bash
   pip install -r requirements.txt
   ```

3. Télécharger et lancer le modèle Ollama utilisé (mistral) :

   ```bash
   ollama run mistral
   ```

   Laisse ce terminal ouvert (ou vérifie qu'Ollama tourne déjà en service avec
   `ollama list`).

## Pour lancer

```bash
streamlit run app.py
```

`http://localhost:8501`.

Utilisation

- Charger des documents :dépose un ou plusieurs
   fichiers PDF, TXT ou MD.
- Indexer** : clique sur " Indexer les documents". Un message de succès
   confirme que la base vectorielle est prête.
- **Choisir le mode** via le toggle "Activer l'assistant RAG (LLM)" :
   - **Désactivé** → Mode *recherche sémantique pure* : affiche les extraits
     bruts les plus proches de la question, sans génération de texte.
   - **Activé** → Mode *RAG complet* : le LLM local (Ollama / mistral) génère
     une réponse rédigée, strictement basée sur les extraits récupérés, avec
     les sources consultables dans un menu dépliable.
-Poser une question** dans la barre de chat en bas.

>>>>>>> f117a08 (put code in repot git)

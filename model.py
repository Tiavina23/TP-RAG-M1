"""
model.py
--------
Tout ce qui concerne les MODÈLES et la configuration globale :
- constantes de configuration (tailles de chunks, chemins, nom des modèles)
- création de l'instance d'embeddings
- création de l'instance du LLM (Ollama)
"""

import os
import tempfile
import uuid

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import Ollama


# =====================================================================
# CONFIGURATION GLOBALE
# =====================================================================

# Modèle d'embeddings HuggingFace (local, léger, rapide)
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Nom du modèle LLM servi par Ollama (doit être déjà "pull" localement)
OLLAMA_MODEL_NAME = "mistral"

# Paramètres de découpage (chunking) du texte
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

# Nombre d'extraits (chunks) récupérés lors d'une recherche
TOP_K = 3


def new_persist_dir():
    """
    Retourne un chemin de dossier temporaire UNIQUE pour la base Chroma.
    Un nouveau dossier est créé à chaque indexation plutôt que de réutiliser
    toujours le même chemin : cela évite les erreurs 'readonly database'
    causées par un ancien dossier resté verrouillé après un plantage ou une
    fermeture brutale de l'application.
    """
    return os.path.join(tempfile.gettempdir(), f"rag_tp_chroma_db_{uuid.uuid4().hex}")


# =====================================================================
# INSTANCIATION DES MODÈLES
# =====================================================================

def get_embeddings():
    """Retourne l'instance du modèle d'embeddings HuggingFace."""
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)


def get_llm():
    """Retourne l'instance du LLM local servi par Ollama."""
    return Ollama(model=OLLAMA_MODEL_NAME)
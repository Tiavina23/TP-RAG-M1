"""
functions.py
------------
Toutes les FONCTIONS de la logique métier de l'application :
- gestion de l'état de session Streamlit
- Étape 2 : pipeline d'ingestion (extraction, chunking, vectorisation)
- Étape 3 : recherche sémantique pure
- Étape 4 : RAG complet (prompt + appel LLM)
"""

import os
import tempfile

import streamlit as st
from langchain_community.document_loaders import PyMuPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import PromptTemplate

from model import CHUNK_SIZE, CHUNK_OVERLAP, TOP_K, get_embeddings, new_persist_dir


# =====================================================================
# ÉTAT DE SESSION STREAMLIT
# =====================================================================

def init_session_state():
    """Initialise les variables persistantes de la session Streamlit."""
    if "vectorstore" not in st.session_state:
        st.session_state.vectorstore = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []  # liste de dicts {"role": ..., "content": ...}
    if "indexed_files" not in st.session_state:
        st.session_state.indexed_files = []  # noms des fichiers déjà indexés


# =====================================================================
# ÉTAPE 2 : PIPELINE D'INGESTION
# =====================================================================

def load_document(uploaded_file):
    """
    Sauvegarde temporairement le fichier uploadé sur disque puis le charge
    avec le DocumentLoader LangChain adapté à son extension.

    Retourne une liste d'objets Document (avec leur métadonnée 'source').
    """
    suffix = os.path.splitext(uploaded_file.name)[1].lower()

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
        tmp_file.write(uploaded_file.getbuffer())
        tmp_path = tmp_file.name

    try:
        if suffix == ".pdf":
            loader = PyMuPDFLoader(tmp_path)
        elif suffix in (".txt", ".md"):
            loader = TextLoader(tmp_path, encoding="utf-8")
        else:
            raise ValueError(f"Format de fichier non supporté : {suffix}")

        documents = loader.load()

        for doc in documents:
            doc.metadata["source"] = uploaded_file.name

        return documents
    finally:
        os.remove(tmp_path)


def index_documents(uploaded_files):
    """
    Pipeline complet d'ingestion :
      1. Extraction du texte de chaque fichier
      2. Découpage en chunks
      3. Vectorisation (embeddings) et stockage dans ChromaDB

    Retourne le vectorstore Chroma résultant (ou None si aucun texte extrait).
    """
    all_documents = []

    for uploaded_file in uploaded_files:
        docs = load_document(uploaded_file)
        all_documents.extend(docs)

    if not all_documents:
        return None

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = text_splitter.split_documents(all_documents)

    embeddings = get_embeddings()

    # Nouveau dossier de persistance à chaque indexation, pour éviter les
    # conflits avec un ancien dossier resté verrouillé (voir new_persist_dir).
    persist_dir = new_persist_dir()

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_dir,
    )

    return vectorstore


# =====================================================================
# ÉTAPE 3 : MODE "RECHERCHE SÉMANTIQUE PURE"
# =====================================================================

def semantic_search(vectorstore, query, k=TOP_K):
    """
    Interroge la base vectorielle et retourne les k documents (chunks)
    les plus proches sémantiquement de la requête, sans appel à un LLM.
    """
    return vectorstore.similarity_search(query, k=k)


# =====================================================================
# ÉTAPE 4 : MODE "RAG COMPLET"
# =====================================================================

# Prompt système strict : on ordonne explicitement au LLM de ne répondre
# qu'à partir du contexte fourni
RAG_PROMPT_TEMPLATE = """Tu es un assistant qui répond STRICTEMENT à partir du contexte fourni ci-dessous, extrait des documents de l'utilisateur.
Règles impératives :
- Utilise UNIQUEMENT les informations présentes dans le contexte pour répondre.
- Si la réponse ne se trouve pas dans le contexte, réponds explicitement que tu ne trouves pas l'information dans les documents fournis, sans inventer.
- Réponds de manière claire et concise, en français.

Contexte :
{context}

Question : {question}

Réponse :"""

RAG_PROMPT = PromptTemplate(
    template=RAG_PROMPT_TEMPLATE,
    input_variables=["context", "question"],
)


def general_chat_answer(llm, query):
    """
    Réponse "hors RAG" : utilisée uniquement quand aucun document n'est
    encore indexé mais que le mode LLM est activé. Le modèle répond
    librement (sans contexte documentaire), utile pour un simple
    "bonjour" ou une question générale avant d'avoir chargé des fichiers.
    """
    return llm.invoke(query)


def rag_answer(vectorstore, llm, query, k=TOP_K):
    """
    Pipeline RAG complet :
      1. Récupère les chunks pertinents (comme en mode recherche sémantique)
      2. Construit le prompt avec le contexte injecté
      3. Interroge le LLM local et retourne la réponse + les sources
    """
    retrieved_docs = vectorstore.similarity_search(query, k=k)

    context_text = "\n\n---\n\n".join(
        f"[Source: {doc.metadata.get('source', 'inconnue')}]\n{doc.page_content}"
        for doc in retrieved_docs
    )

    prompt = RAG_PROMPT.format(context=context_text, question=query)
    answer = llm.invoke(prompt)

    return answer, retrieved_docs
"""
app.py
------
Étape 1 : Interface Streamlit + point d'entrée de l'application.
Contient uniquement l'affichage (sidebar + zone de chat) et l'orchestration :
toute la logique est importée depuis functions.py et model.py.

Lancement : streamlit run app.py
"""

import streamlit as st

from model import OLLAMA_MODEL_NAME, get_llm
from functions import (
    init_session_state,
    index_documents,
    semantic_search,
    rag_answer,
    general_chat_answer,
)


def main():
    st.set_page_config(page_title="RAG Local - Clone NotebookLM", page_icon="📚")
    st.title("📚 Assistant RAG")
    st.caption("Chargez vos documents, indexez-les, puis interrogez-les — 100% local.")

    init_session_state()

    # ------------------ BARRE LATÉRALE ------------------
    with st.sidebar:
        st.header("⚙️ Gestion des documents")

        uploaded_files = st.file_uploader(
            "Déposez vos fichiers (PDF, TXT, MD)",
            type=["pdf", "txt", "md"],
            accept_multiple_files=True,
        )

        if st.button("🔍 Indexer les documents", use_container_width=True):
            if not uploaded_files:
                st.warning("Merci de charger au moins un fichier avant d'indexer.")
            else:
                with st.spinner("Extraction, découpage et vectorisation en cours..."):
                    vectorstore = index_documents(uploaded_files)
                if vectorstore is not None:
                    st.session_state.vectorstore = vectorstore
                    st.session_state.indexed_files = [f.name for f in uploaded_files]
                    st.success(f"{len(uploaded_files)} document(s) indexé(s) avec succès !")
                else:
                    st.error("Aucun texte n'a pu être extrait des fichiers fournis.")

        if st.session_state.indexed_files:
            st.markdown("**📄 Documents indexés :**")
            for name in st.session_state.indexed_files:
                st.markdown(f"- {name}")

        st.divider()

        st.header("Mode de fonctionnement")
        llm_mode = st.toggle(
            "Activer l'assistant RAG (LLM)",
            value=False,
            help="Désactivé = recherche sémantique brute. Activé = réponse générée par le LLM local.",
        )

        if llm_mode:
            st.caption(f"Modèle Ollama utilisé : `{OLLAMA_MODEL_NAME}`")

        if st.button("🗑️ Réinitialiser la conversation", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()

    # ------------------ ZONE DE CONVERSATION ------------------

    # Affichage de l'historique
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("sources"):
                with st.expander("📎 Extraits sources utilisés"):
                    for i, src in enumerate(message["sources"], start=1):
                        st.markdown(
                            f"**Extrait {i} — source : `{src['source']}`**\n\n"
                            f"> {src['content']}"
                        )

    user_query = st.chat_input("Posez une question sur vos documents...")

    if user_query:
        # On affiche et on enregistre TOUJOURS le message utilisateur,
        # même si aucun document n'est encore indexé.
        st.session_state.chat_history.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)

        with st.chat_message("assistant"):
            if st.session_state.vectorstore is None and not llm_mode:
                # Mode "recherche sémantique pure" sans documents indexés :
                # il n'y a littéralement rien à chercher, donc on informe
                # clairement l'utilisateur (message conservé dans l'historique,
                # contrairement à l'ancien st.stop() qui coupait tout et
                # donnait l'impression que l'appli ne répondait pas).
                response_text = (
                    "Je ne peux pas encore chercher dans vos documents : aucun fichier "
                    "n'est indexé. Merci de charger au moins un fichier (PDF, TXT ou MD) "
                    "dans la barre latérale, puis de cliquer sur **🔍 Indexer les documents**. "
                    "Astuce : si vous voulez juste discuter en attendant, activez le mode "
                    "'Assistant RAG (LLM)' ci-contre."
                )
                st.warning(response_text)
                sources = []

            elif st.session_state.vectorstore is None and llm_mode:
                # Mode LLM activé mais pas encore de documents indexés :
                # on laisse le modèle répondre librement (sans contexte
                # documentaire), pratique pour un simple "bonjour" avant
                # d'avoir chargé des fichiers.
                with st.spinner("Génération de la réponse par le LLM local..."):
                    try:
                        llm = get_llm()
                        answer = general_chat_answer(llm, user_query)
                    except Exception as e:
                        answer = (
                            "Erreur lors de l'appel au modèle Ollama. "
                            f"Vérifiez qu'Ollama tourne bien en local et que le modèle "
                            f"'{OLLAMA_MODEL_NAME}' est disponible.\n\nDétail : {e}"
                        )

                st.info(
                    "ℹ️ Aucun document indexé : réponse générée sans contexte documentaire."
                )
                st.markdown(answer)
                response_text = answer
                sources = []

            elif not llm_mode:
                # ---- MODE RECHERCHE SÉMANTIQUE ----
                with st.spinner("Recherche des extraits les plus pertinents..."):
                    results = semantic_search(st.session_state.vectorstore, user_query)

                if not results:
                    response_text = "Aucun extrait pertinent n'a été trouvé dans les documents."
                    sources = []
                else:
                    response_text = "Voici les extraits les plus proches de votre question :"
                    sources = [
                        {"source": doc.metadata.get("source", "inconnue"), "content": doc.page_content}
                        for doc in results
                    ]

                st.markdown(response_text)
                for i, src in enumerate(sources, start=1):
                    st.markdown(f"**Extrait {i} — source : `{src['source']}`**")
                    st.info(src["content"])

            else:
                # ---- MODE RAG COMPLET ----
                with st.spinner("Génération de la réponse par le LLM local..."):
                    try:
                        llm = get_llm()
                        answer, retrieved_docs = rag_answer(
                            st.session_state.vectorstore, llm, user_query
                        )
                    except Exception as e:
                        answer = (
                            "Erreur lors de l'appel au modèle Ollama. "
                            f"Vérifiez qu'Ollama tourne bien en local et que le modèle "
                            f"'{OLLAMA_MODEL_NAME}' est disponible.\n\nDétail : {e}"
                        )
                        retrieved_docs = []

                response_text = answer
                sources = [
                    {"source": doc.metadata.get("source", "inconnue"), "content": doc.page_content}
                    for doc in retrieved_docs
                ]

                st.markdown(response_text)
                if sources:
                    with st.expander("📎 Extraits sources utilisés"):
                        for i, src in enumerate(sources, start=1):
                            st.markdown(
                                f"**Extrait {i} — source : `{src['source']}`**\n\n"
                                f"> {src['content']}"
                            )

        st.session_state.chat_history.append(
            {"role": "assistant", "content": response_text, "sources": sources}
        )


if __name__ == "__main__":
    main()
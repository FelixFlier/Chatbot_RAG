import streamlit as st
import os
import logging
from typing import Dict, List, Any
from pathlib import Path
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import GoogleGenerativeAI
import pickle

# Import eigene Module
from styles import apply_enhanced_custom_css
from ui_utils import (
    display_chat_message,
    display_typical_questions,
    display_metadata,
    add_scroll_button,
    add_font_size_control
)
from callbacks import ProcessCallback
from config import load_embeddings, load_llm, load_vector_stores, VECTOR_STORE_PATH
from session_state import initialize_session_state
from chatbot_new import IntelligentRAGChatbot, InitializationError, ChatbotError
import streamlit as st
from pathlib import Path
from urllib.parse import urlencode  # Import für urlencode
import random  # Import für random
import string  # Impor

# Logging Setup
logger = logging.getLogger(__name__)

# Konstanten
BASE_PATH = Path(__file__).resolve().parent
VECTOR_STORE_PATH = BASE_PATH / "vector_stores"
VECTOR_STORE_FILE = VECTOR_STORE_PATH / "vectorstore.pkl"

# Cache Funktionen
@st.cache_resource
def load_embeddings():
    try:
        return HuggingFaceEmbeddings(
            model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
        )
    except Exception as e:
        logger.error(f"Fehler beim Laden der Embeddings: {e}")
        raise

@st.cache_resource
def load_llm():
    try:
        return GoogleGenerativeAI(
            model="gemini-pro",
            temperature=0.4,
            top_p=0.95,
            max_output_tokens=700
        )
    except Exception as e:
        logger.error(f"Fehler beim Laden des LLM: {e}")
        raise

@st.cache_data
def load_vector_stores():
    try:
        with open(VECTOR_STORE_PATH / "vectorstore.pkl", 'rb') as f:
            vectorstore = pickle.load(f)
        with open(VECTOR_STORE_PATH / "bm25.pkl", 'rb') as f:
            bm25 = pickle.load(f)
        return vectorstore, bm25
    except Exception as e:
        logger.error(f"Fehler beim Laden der Vector Stores: {e}")
        raise

def handle_response(response: Dict[str, Any], message_index: int):
    """Verarbeitet die Chatbot-Antwort mit Metadaten"""
    try:
        if response.get('error'):
            st.error(response['error'])
            return

        answer_text = response.get('response', '')
        if not answer_text:
            st.error("Keine Antwort vom Chatbot erhalten")
            return

        metadata = response.get('metadata', {})

        # Füge Nachricht mit Metadaten zum State hinzu
        st.session_state.messages.append({
            "role": "assistant",
            "content": answer_text,
            "metadata": metadata  # Stelle sicher, dass die Metadaten gespeichert werden
        })

        # Zeige Nachricht und Metadaten an
        display_chat_message({
            "role": "assistant", 
            "content": answer_text,
            "metadata": metadata
        }, message_index)

    except Exception as e:
        logger.error(f"Fehler bei der Antwortverarbeitung: {e}")
        st.error("Die Antwort konnte nicht verarbeitet werden.")

def process_chat_interface():
    """Verarbeitet die Chat-Oberfläche"""
    try:
        # Zeige bestehende Nachrichten
        for idx, message in enumerate(st.session_state.messages):
            display_chat_message(message, idx)

        # Zeige typische Fragen wenn keine Interaktion
        if len(st.session_state.messages) <= 1:
            typical_questions = [
                "Wofür steht die Corporate Governance?",
                "Was ist der Ablauf einer internen Revision?",
                "Was sind die Dimensionen der Nachhaltigkeit?",
                "Was sind die Aufgaben eines Wirtschaftsprüfers?"
            ]
            display_typical_questions(typical_questions)

        # Chat Eingabe
        if prompt := st.chat_input("Ihre Frage...", key="chat_input"):
            process_user_input(prompt)
        elif "user_prompt" in st.session_state:
            # Verarbeite die Frage aus dem Session State (von den Typical Questions Buttons)
            process_user_input(st.session_state.user_prompt)
            del st.session_state.user_prompt
        
        # Kopieren von Text aus der Zwischenablage
        if 'to_copy' in st.session_state:
            st.code(st.session_state.to_copy, language="markdown")
            st.success("Text wurde in die Zwischenablage kopiert!")
            del st.session_state.to_copy

    except Exception as e:
        logger.error(f"Fehler im Chat Interface: {e}")
        st.error("Fehler in der Chat-Oberfläche")

def process_user_input(prompt: str):
    """Verarbeitet Benutzereingaben"""
    try:
        # Füge Benutzernachricht hinzu
        st.session_state.messages.append({"role": "user", "content": prompt})
        display_chat_message({"role": "user", "content": prompt}, len(st.session_state.messages)-1)

        # Aktualisiere Konversationskontext
        st.session_state.conversation_context["interaction_count"] += 1

        # Verarbeite Anfrage
        with st.spinner("Verarbeite Anfrage..."):
            callback = ProcessCallback(st.empty())
            
            # Hole Antwort vom Chatbot
            response = st.session_state.chatbot.get_response(
                query=prompt, 
                callback=callback
            )

            # Update Konversationskontext
            if response.get('response'):
                st.session_state.conversation_context["last_response"] = response['response']

            handle_response(response, len(st.session_state.messages))

    except Exception as e:
        logger.error(f"Fehler bei der Eingabeverarbeitung: {e}")
        st.error("Ihre Anfrage konnte nicht verarbeitet werden.")
def display_feedback_message():
    """Zeigt Feedback-Nachrichten mit angepassten Farben an"""
    if 'feedback' in st.session_state:
        if st.session_state.feedback['feedbackType'] == 'positive':
            st.success("Vielen Dank für Ihr positives Feedback!", icon="👍")
        elif st.session_state.feedback['feedbackType'] == 'negative':
            st.warning("Vielen Dank für Ihr Feedback. Wir werden es prüfen!", icon="👎")
        del st.session_state.feedback

def add_theme_selection():
    """Fügt Theme-Auswahl mit eindeutigen Keys hinzu"""
    st.markdown("### Design")
    cols = st.columns(2)
    
    # Speichere vorheriges Theme
    previous_theme = st.session_state.get("theme", "light")
    
    with cols[0]:
        if st.button("🌞 Hell", key="light_theme_button", use_container_width=True):
            st.session_state.theme = "light"
            if previous_theme != "light":
                st.rerun()
            
    with cols[1]:
        if st.button("🌙 Dunkel", key="dark_theme_button", use_container_width=True):
            st.session_state.theme = "dark"
            if previous_theme != "dark":
                st.rerun()

def main():
    st.set_page_config(
        page_title="KI-Assistent für Wirtschaftsprüfungsthemen",
        page_icon="🤖",
        layout="centered",
        initial_sidebar_state="expanded"
    )
    
    # State Initialisierung
    initialize_session_state()
    
    # CSS anwenden
    apply_enhanced_custom_css()
    
    # Sidebar
    with st.sidebar:
        st.markdown("## Einstellungen")
        add_font_size_control()
        st.markdown("---")
        # Theme Selection - nur EINMAL aufrufen
        add_theme_selection()
        st.markdown("---")
        st.markdown("## Über den KI-Assistenten")
        st.markdown(
            """
            Entwickelt, um Fragen zu Wirtschaftsprüfungsthemen zu beantworten.
            Nutzt fortschrittliche Technologien, um relevante Informationen zu finden.

            **Hinweis:** Die Antworten dienen nur zu Informationszwecken und ersetzen keine professionelle Beratung.
            """
        )
        st.markdown("---")

    # Feedback Anzeige
    if st.session_state.get('feedback'):
        display_feedback_message()

    # Hauptbereich
    st.title("KI-Assistent für Wirtschaftsprüfungsthemen")
    process_chat_interface()

    # Scroll Button
    if len(st.session_state.messages) > 5:
        add_scroll_button()

if __name__ == "__main__":
    try:
        os.environ["GOOGLE_API_KEY"] = API_Key  # Replace with your actual API key
        main()
    except Exception as e:
        st.error(f"Kritischer Fehler: {str(e)}")

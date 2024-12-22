# session_state.py
import streamlit as st
import logging
from pathlib import Path
from typing import Optional
from chatbot_new import InitializationError, StateError
from config import VECTOR_STORE_PATH
from chatbot_new import IntelligentRAGChatbot
import os

logger = logging.getLogger(__name__)

def initialize_session_state():
    """Initialisiert den Session State"""
    try:
        if "theme" not in st.session_state:
            st.session_state.theme = "light"
            
        _initialize_basic_state()
        
        if "chatbot" not in st.session_state:
            with st.spinner("Initialisiere Chatbot..."):
                _initialize_chatbot()
    except Exception as e:
        _handle_initialization_error(e)

def _initialize_basic_state():
    """Initialisiert die grundlegenden Session State Variablen"""
    try:
        # Willkommensnachricht
        if "messages" not in st.session_state:
            st.session_state.messages = [{"role": "assistant", "content": "Hallo! Ich bin Ihr KI-Assistent für Wirtschaftsprüfungsthemen. Wie kann ich Ihnen helfen?"}]

        # Basis-Einstellungen
        default_settings = {
            "feedback": None,
            "debug": False,
            "theme": "light",
            "error_count": 0,
            "conversation_context": {  # Neu: Konversationskontext
                "last_response": None,
                "active_themes": [],
                "interaction_count": 0
            }
        }

        for key, value in default_settings.items():
            if key not in st.session_state:
                st.session_state[key] = value

    except Exception as e:
        raise StateError(f"Fehler bei Basis-Initialisierung: {str(e)}")

def _initialize_chatbot():
    """Initialisiert den Chatbot mit Fehlerprüfung"""
    try:
        # Pfadprüfungen
        _verify_paths()

        # API Key Prüfung
        api_key = _verify_api_key()

        # Chatbot Initialisierung
        logger.info("Starte Chatbot-Initialisierung...")
        st.session_state.chatbot = IntelligentRAGChatbot(
            vectorstore_file=VECTOR_STORE_PATH / "vectorstore.pkl",
            bm25_file=VECTOR_STORE_PATH / "bm25.pkl",
            api_key=api_key
        )

    except Exception as e:
        raise InitializationError(f"Chatbot-Initialisierung fehlgeschlagen: {str(e)}")

def _verify_paths():
    """Überprüft die erforderlichen Dateipfade"""
    if not VECTOR_STORE_PATH.exists():
        raise FileNotFoundError(f"Verzeichnis nicht gefunden: {VECTOR_STORE_PATH}")

    required_files = {
        "vectorstore.pkl": "Vector Store Datei",
        "bm25.pkl": "BM25 Datei"
    }

    for filename, description in required_files.items():
        file_path = VECTOR_STORE_PATH / filename
        if not file_path.exists():
            raise FileNotFoundError(f"{description} nicht gefunden: {file_path}")

def _verify_api_key() -> str:
    """Überprüft und gibt den API Key zurück"""
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("Google API Key nicht gefunden. Bitte setzen Sie die Umgebungsvariable.")
    return api_key

def _handle_initialization_error(error: Exception):
    """Zentrale Fehlerbehandlung für Initialisierungsfehler"""
    error_messages = {
        FileNotFoundError: "Datei nicht gefunden",
        ValueError: "Konfigurationsfehler",
        InitializationError: "Chatbot-Initialisierungsfehler",
        StateError: "Session State Fehler"
    }

    error_type = type(error)
    base_message = error_messages.get(error_type, "Unerwarteter Fehler")

    error_msg = f"{base_message}: {str(error)}"

    # Debug-Modus Handling
    if st.session_state.get('debug', False):
        st.error(f"Detaillierter Fehler: {error_msg}")
        logger.error(error_msg, exc_info=True)
    else:
        st.error("Ein Fehler ist bei der Initialisierung aufgetreten.")
        logger.error(error_msg)

    # Fehlerzähler
    st.session_state.error_count = st.session_state.get('error_count', 0) + 1

    if st.session_state.error_count > 3:
        st.error("Mehrere Fehler aufgetreten. Bitte laden Sie die Seite neu.")
        logger.critical("Kritische Fehlerhäufung bei der Initialisierung")

    raise error

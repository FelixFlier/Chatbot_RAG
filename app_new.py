import streamlit as st
import os
import pickle
from typing import Dict, List, Any
from chatbot_new import IntelligentRAGChatbot, VECTOR_STORE_PATH, InitializationError
from callbacks import ProcessCallback
from ui_utils import (
    apply_enhanced_custom_css,
    add_copy_button,
    add_scroll_button,
    add_font_size_control,
    display_metadata,
    initialize_session_state,
    add_modern_feedback,
    handle_conversation_flow
)
from dotenv import load_dotenv
load_dotenv()

def check_pickle_files():
    """Überprüft das Vorhandensein und den Inhalt der Pickle-Dateien"""
    print("\nÜberprüfe Pickle-Dateien...")
    files = ['vectorstore.pkl', 'bm25.pkl', 'documents.pkl']
    for file in files:
        full_path = os.path.join(VECTOR_STORE_PATH, file)
        print(f"\nPrüfe Datei: {full_path}")
        print(f"Datei existiert: {os.path.exists(full_path)}")
        if os.path.exists(full_path):
            try:
                with open(full_path, 'rb') as f:
                    data = pickle.load(f)
                    print(f"Typ: {type(data)}")
                    if hasattr(data, '__len__'):
                        print(f"Länge: {len(data)}")
            except Exception as e:
                print(f"Fehler beim Laden von {file}: {str(e)}")
        else:
            print(f"Datei {file} fehlt!")
def handle_response(response: Dict[str, Any], message_index: int):
    """
    Verarbeitet die Chatbot-Antwort und zeigt sie an
    
    Args:
        response: Die Antwort des Chatbots
        message_index: Index der Nachricht
    """
    if response.get('error'):
        st.error(response['error'])
        return

    answer = response['answer']
    metadata = response.get('metadata', {})
    
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "metadata": metadata
    })
    
    with st.chat_message("assistant"):
        st.markdown(answer)
        add_copy_button(answer, key=message_index)
        if metadata:
            display_metadata(metadata)
        add_modern_feedback(message_index)

def handle_error(error: Exception):
    """
    Verarbeitet Fehler in der Chatbot-Verarbeitung
    
    Args:
        error: Die aufgetretene Exception
    """
    error_message = f"Es ist ein Fehler aufgetreten: {str(error)}"
    st.error(error_message)
    st.session_state.messages.append({
        "role": "assistant",
        "content": error_message
    })

def main():
    st.set_page_config(
        page_title="KI-Assistent für Wirtschaftsprüfungsthemen",
        page_icon="🤖",
        layout="centered",
        initial_sidebar_state="expanded"
    )
    
    add_font_size_control()
    apply_enhanced_custom_css()
    initialize_session_state()
    
    st.title("KI-Assistent für Wirtschaftsprüfungsthemen")
    
    # Chat Verlauf anzeigen
    for idx, message in enumerate(st.session_state.messages):
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("metadata"):
                display_metadata(message["metadata"])
            if message["role"] == "assistant":
                add_modern_feedback(idx)
    
    # Chat Input und Verarbeitung
    if prompt := st.chat_input("Ihre Frage..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Prüfe erst ob die Konversation beendet werden soll
        if handle_conversation_flow(st):
            farewell_message = (
                "Vielen Dank für das Gespräch! Ich hoffe, ich konnte Ihnen weiterhelfen. "
                "Falls Sie später weitere Fragen haben, können Sie sich jederzeit wieder an mich wenden."
            )
            st.session_state.messages.append({
                "role": "assistant",
                "content": farewell_message
            })
            with st.chat_message("assistant"):
                st.markdown(farewell_message)
            return
        
        status_container = st.empty()
        callback = ProcessCallback(status_container)
        
        try:
            response = st.session_state.chatbot.get_response(prompt, callback)
            
            if isinstance(response, dict):
                handle_response(response, len(st.session_state.messages))
            
        except Exception as e:
            handle_error(e)
        
        finally:
            if status_container:
                status_container.empty()
    
    # Scroll-Button bei vielen Nachrichten
    if len(st.session_state.messages) > 5:
        add_scroll_button()

if __name__ == "__main__":
    try:
        if not os.path.exists(VECTOR_STORE_PATH):
            os.makedirs(VECTOR_STORE_PATH)
        api_key = st.secrets["GOOGLE_API_KEY"]
        os.environ["GOOGLE_API_KEY"] = api_key
        main()
        
    except Exception as e:
        st.error(f"Kritischer Fehler: {str(e)}")


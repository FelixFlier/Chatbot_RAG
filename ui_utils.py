from datetime import datetime
import streamlit as st
import os
import pickle
from typing import Dict, Any
from chatbot_new import IntelligentRAGChatbot, InitializationError


# Am Anfang der Datei die Konstanten hinzufügen:
BASE_PATH = os.path.dirname(os.path.abspath(__file__))
VECTOR_STORE_PATH = os.path.join(BASE_PATH, "vector_stores")

# Diese Funktionen müssen VOR der IntelligentRAGChatbot-Klasse platziert werden
def apply_enhanced_custom_css():
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap');

        /* Color Variables */
        :root {
            --primary: #06b6d4;
            --primary-light: #22d3ee;
            --primary-dark: #0891b2;
            --background-dark: #0a1930;
            --background-light: #1a1f3c;
            --surface: rgba(30, 27, 75, 0.5);
            --text-primary: #f8fafc;
            --text-secondary: #cbd5e1;
        }

        /* Base Styles & Navigation Bar Fix */
        .stApp {
            background: linear-gradient(145deg, var(--background-dark) 0%, #164e63 100%) !important;
            color: var(--text-primary);
            font-family: 'Plus Jakarta Sans', sans-serif;
            line-height: 1.7;
        }

        /* Remove default header */
        header[data-testid="stHeader"] {
            display: none;
        }

        /* Clean up sidebar */
        section[data-testid="stSidebar"] {
            background: linear-gradient(180deg, var(--background-dark) 0%, #164e63 100%) !important;
            border-right: 1px solid rgba(6, 182, 212, 0.1);
        }

        .stSidebar > div:first-child {
            padding-top: 2rem;
        }

        /* Typography */
        h1 {
            color: var(--text-primary);
            text-align: center;
            font-size: 2.5rem;
            font-weight: 700;
            margin-bottom: 2.5rem;
            letter-spacing: -0.02em;
            background: linear-gradient(135deg, #22d3ee 0%, #06b6d4 50%, #0891b2 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }

        /* Chat Messages */
        .stChatMessage {
            background: rgba(30, 27, 75, 0.3) !important;
            backdrop-filter: blur(12px);
            border: 1px solid rgba(6, 182, 212, 0.1);
            border-radius: 16px;
            padding: 1.5rem;
            margin: 1.5rem 0;
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.15);
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        }

        /* Message Layout */
        .stChatMessage[data-testid="user-message"] {
            background: rgba(6, 182, 212, 0.1) !important;
            margin-left: 12%;
            margin-right: 2%;
            border-left: 3px solid var(--primary);
        }

        .stChatMessage[data-testid="assistant-message"] {
            background: rgba(10, 25, 48, 0.7) !important;
            margin-right: 12%;
            margin-left: 2%;
            border-right: 3px solid var(--primary);
        }

        /* Enhanced Input Area */
        .stChatInput {
            background-color: transparent !important;
        }

        .stChatInput > div {
            background-color: transparent !important;
        }

        div[data-testid="stChatInput"] {
            background-color: transparent !important;
        }

        .stTextInput > div {
            background-color: transparent !important;
        }

        div[data-testid="textInput"] {
            background-color: transparent !important;
        }

        .stTextInput > div > div {
            background-color: transparent !important;
        }

        .stTextInput > div > div > input {
            background: rgba(10, 25, 48, 0.7) !important;
            color: var(--text-primary);
            border: 1px solid rgba(6, 182, 212, 0.3);
            border-radius: 12px;
            padding: 1rem 1.5rem;
            font-size: 1rem;
            font-weight: 500;
            transition: all 0.2s ease;
            margin-bottom: 1rem;
            width: 100%;
        }

        .stTextInput > div > div > input:focus {
            border-color: var(--primary);
            box-shadow: 0 0 0 3px rgba(6, 182, 212, 0.15);
        }

        /* Chat Input Container Fix */
        .stChatInputContainer {
            background-color: transparent !important;
            border: none !important;
            padding: 0 !important;
        }

        section[data-testid="stChatInput"] {
            background-color: transparent !important;
            border: none !important;
            padding: 0 !important;
        }

        /* Source Details */
        .streamlit-expanderHeader {
            background: rgba(6, 182, 212, 0.08);
            border-radius: 10px;
            padding: 1rem;
            border: 1px solid rgba(6, 182, 212, 0.15);
            transition: all 0.2s ease;
        }

        /* Copy Button */
        button[data-testid="baseButton-secondary"] {
            background: rgba(6, 182, 212, 0.1);
            color: var(--primary-light);
            border: 1px solid rgba(6, 182, 212, 0.2);
            border-radius: 8px;
            padding: 0.5rem 1rem;
            font-weight: 500;
            transition: all 0.2s ease;
        }

        button[data-testid="baseButton-secondary"]:hover {
            background: rgba(6, 182, 212, 0.2);
            border-color: var(--primary);
        }

        /* Footer Area */
        footer {
            visibility: hidden;
        }

        /* Sidebar Text Size Control */
        .stSlider {
            padding: 1rem;
        }

        .stSlider > div > div > div {
            background-color: var(--primary) !important;
        }

        /* Remove Bottom Space */
        .main .block-container {
            padding-bottom: 2rem;
            max-width: 1000px;
        }

        /* Custom Scrollbar */
        ::-webkit-scrollbar {
            width: 8px;
            height: 8px;
        }

        ::-webkit-scrollbar-track {
            background: rgba(10, 25, 48, 0.1);
        }

        ::-webkit-scrollbar-thumb {
            background: var(--primary);
            border-radius: 4px;
        }

        ::-webkit-scrollbar-thumb:hover {
            background: var(--primary-light);
        }
        </style>
        """,
        unsafe_allow_html=True
    )

def add_copy_button(text: str, key: str):
    if st.button("📋 Kopieren", key=f"copy_{key}"):
        st.session_state[f"copied_{key}"] = True
        st.success("Text kopiert!")

def add_scroll_button():
    st.markdown("""
        <div class="scroll-button" onclick="window.scrollTo(0,document.body.scrollHeight);">
            ⬇
        </div>
    """, unsafe_allow_html=True)

def add_font_size_control():
    st.sidebar.markdown("### Textgröße")
    font_size = st.sidebar.slider("Wählen Sie die Textgröße", 12, 24, 16)
    st.markdown(f"""
        <style>
        .stChatMessage p {{
            font-size: {font_size}px !important;
        }}
        </style>
    """, unsafe_allow_html=True)

def check_pickle_files():
    files = ['vectorstore.pkl', 'bm25.pkl', 'documents.pkl']
    for file in files:
        full_path = os.path.join(VECTOR_STORE_PATH, file)
        try:
            with open(full_path, 'rb') as f:
                data = pickle.load(f)
                print(f"\nDebug {file}:")
                print(f"Typ: {type(data)}")
                if hasattr(data, '__len__'):
                    print(f"Länge: {len(data)}")
        except Exception as e:
            print(f"Fehler beim Laden von {file}: {str(e)}")

def display_processing_status(status: str, progress: float = 0.0):
    """Zeigt Verarbeitungsstatus mit Fortschrittsbalken"""
    progress_container = st.empty()
    with progress_container:
        progress_bar = st.progress(progress)
        status_text = st.empty()
        status_text.write(status)
    return progress_container

def add_modern_feedback(message_key: str):
    """Fügt ein modernes, hover-basiertes Feedback-System hinzu"""
    st.markdown(f"""
        <style>
        .feedback-container-{message_key} {{
            position: absolute;
            right: -60px;
            top: 50%;
            transform: translateY(-50%);
            opacity: 0;
            transition: opacity 0.3s ease;
            background: rgba(255, 255, 255, 0.1);
            padding: 8px;
            border-radius: 8px;
            display: flex;
            flex-direction: column;
            gap: 8px;
        }}
        
        .message-container-{message_key}:hover .feedback-container-{message_key} {{
            opacity: 1;
        }}
        
        .feedback-button {{
            background: transparent;
            border: none;
            cursor: pointer;
            padding: 4px;
            transition: transform 0.2s ease;
            color: #cbd5e1;
        }}
        
        .feedback-button:hover {{
            transform: scale(1.2);
            color: #06b6d4;
        }}
        
        .message-container-{message_key} {{
            position: relative;
            padding-right: 60px;
        }}
        </style>
        
        <div class="message-container-{message_key}">
            <div class="feedback-container-{message_key}">
                <button class="feedback-button" onclick="handleFeedback('positive', '{message_key}')">
                    👍
                </button>
                <button class="feedback-button" onclick="handleFeedback('negative', '{message_key}')">
                    👎
                </button>
            </div>
        </div>
        
        <script>
        function handleFeedback(type, key) {{
            // Hier könnte man AJAX-Calls für die Feedback-Verarbeitung implementieren
            const button = event.target;
            button.style.color = '#06b6d4';
            
            // Feedback in session state speichern
            window.parent.postMessage({{
                type: 'feedback',
                messageKey: key,
                feedbackType: type
            }}, '*');
        }}
        </script>
    """, unsafe_allow_html=True)

def display_metadata(metadata: Dict[str, Any]):
    """Erweiterte Metadaten-Anzeige"""
    with st.expander("🔍 Qualitätsmetriken", expanded=True):
        # Zeige Failed Metrics
        if metadata.get('failed_quality_check'):
            st.warning("Die Antwort hat die Qualitätsanforderungen nicht erfüllt:")
            failed_metrics = metadata.get('failed_metrics', [])
            for metric in failed_metrics:
                st.error(f"- {metric}")
        
        # Zeige Qualitätsmetriken
        if metadata.get('quality_metrics') and metadata['quality_metrics'].get('metrics'):
            metrics = metadata['quality_metrics']['metrics']
            cols = st.columns(4)
            
            def get_color(value: float) -> str:
                if value >= 0.8: return "green"
                if value >= 0.6: return "orange"
                return "red"
            
            metric_names = {
                'relevance_score': 'Relevanz',
                'completeness_score': 'Vollständigkeit',
                'coherence_score': 'Kohärenz',
                'accuracy_score': 'Genauigkeit'
            }
            
            for col, (metric, display_name) in zip(cols, metric_names.items()):
                with col:
                    value = metrics.get(metric, 0)
                    st.markdown(f"""
                        <div style='color: {get_color(value)}'>
                            {display_name}: {value:.2f}
                        </div>
                    """, unsafe_allow_html=True)
        
        # Zeige Kontext-Abdeckung
        if metadata.get('context_coverage') is not None:
            st.write(f"Kontext-Abdeckung: {metadata['context_coverage']:.2%}")
            
def handle_conversation_flow(st):
    """Vereinfachte Konversationssteuerung mit natürlicher Verneinungserkennung"""
    
    rejection_patterns = [
        "nein danke", "keine weiteren fragen", "das war's danke", "das wars danke",
        "nein das wäre alles", "das ist alles danke", "danke das wärs",
        "nein für heute reicht es", "das reicht danke", "danke das genügt",
        "nein das ist alles", "das war alles", "keine fragen mehr", "nein", "nein"
    ]
    
    def is_rejection(text: str) -> bool:
        """Überprüft, ob eine Antwort eine höfliche Verneinung enthält"""
        text = text.lower().strip()
        return any(pattern in text for pattern in rejection_patterns)
    
    try:
        if "messages" in st.session_state and st.session_state.messages:
            last_message = st.session_state.messages[-1]
            if last_message["role"] == "user":
                if is_rejection(last_message["content"]):
                    farewell_message = (
                        "Vielen Dank für das Gespräch! Ich hoffe, ich konnte Ihnen weiterhelfen. "
                        "Falls Sie später weitere Fragen haben, können Sie sich jederzeit wieder an mich wenden."
                    )
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": farewell_message
                    })
                    st.session_state.conversation_ended = True
                    return True
                else:
                    # Wichtig: conversation_ended zurücksetzen, wenn KEINE Verneinung
                    st.session_state.conversation_ended = False

        return False # Diese Zeile ist entscheidend für den weiteren Ablauf
        
    except Exception as e:
        print(f"Fehler in handle_conversation_flow: {str(e)}")
        st.session_state.conversation_ended = False # Auch hier im Fehlerfall zurücksetzen
        return False
                
def initialize_session_state():
    """Erweiterte Session State Initialisierung"""
    if "messages" not in st.session_state:
        st.session_state.messages = [{
            "role": "assistant",
            "content": "Hallo! Ich bin Ihr KI-Assistent für Wirtschaftsprüfungsthemen. Wie kann ich Ihnen helfen?"
        }]
    if "chatbot" not in st.session_state:
        try:
            st.session_state.chatbot = IntelligentRAGChatbot(
                vectorstore_path="vectorstore.pkl",
                bm25_path="bm25.pkl"
            )
        except InitializationError as e:
            st.error(f"Fehler bei der Initialisierung des Chatbots: {str(e)}")
            st.stop()
    
    # Neue States für Feedback und Verarbeitung
    if "feedback_history" not in st.session_state:
        st.session_state.feedback_history = {}
    if "processing_status" not in st.session_state:
        st.session_state.processing_status = None
    if "quality_thresholds" not in st.session_state:
        st.session_state.quality_thresholds = {
            'relevance_score': 0.5,
            'completeness_score': 0.001,
            'coherence_score': 0.5,
            'accuracy_score': 0.5
        }
import streamlit as st
from typing import Dict, Any, List, Union
import logging

logger = logging.getLogger(__name__)

def handle_feedback(feedback_type: str, message_key: str):
    """Verarbeitet Feedback-Interaktionen"""
    st.session_state.feedback = {
        'feedbackType': feedback_type,
        'messageKey': message_key
    }

def display_chat_message(message: Dict[str, Any], idx: int):
    """Zeigt eine Chat-Nachricht mit korrekt angeordneten Buttons und Metadaten an"""
    with st.chat_message(message["role"]):
        # Hauptinhalt
        st.markdown(message["content"])
        
        # Feedback/Copy Buttons für Assistant-Nachrichten
        if message["role"] == "assistant":
            # Buttons nebeneinander in korrekter Reihenfolge
            button_cols = st.columns([0.94, 0.02, 0.02, 0.02])  # Angepasste Spaltenbreiten
            
            with button_cols[1]:  # Kopier-Button links
                if st.button("📋", key=f'copy_{idx}', help="In Zwischenablage kopieren"):
                    st.session_state.to_copy = message["content"]
            with button_cols[2]:  # Negatives Feedback rechts
                if st.button("👎", key=f'thumb_down_{idx}', help="Negatives Feedback"):
                    handle_feedback('negative', idx)
            with button_cols[3]:  # Positives Feedback in der Mitte
                if st.button("👍", key=f'thumb_up_{idx}', help="Positives Feedback"):
                    handle_feedback('positive', idx)
            
            # Zeige Metadaten direkt nach der Nachricht an
            if "metadata" in message:
                display_metadata(message["metadata"])

def display_typical_questions(questions: List[str]):
    """Zeigt typische Fragen als einheitlich gestylte Buttons an"""
    cols = st.columns(len(questions))
    for idx, question in enumerate(questions):
        with cols[idx]:
            # Button mit angepasstem Styling
            if st.button(
                question, 
                key=f'question_{idx}',
                help="Klicken Sie um diese Frage zu stellen",
                use_container_width=True,
                # Styling
                type="secondary"  # Dies stellt das gewünschte Hover-Verhalten wieder her
            ):
                st.session_state.user_prompt = question

def set_chat_input_value(text: str):
    """
    Versucht den Wert der Chat-Eingabe zu setzen (veraltet, nicht mehr verwendet)
    """
    logger.warning("set_chat_input_value wird aufgerufen, ist aber veraltet und sollte nicht mehr verwendet werden.")
    if "chat_input" in st.session_state:
        st.session_state.chat_input = text
    else:
        logger.warning("chat_input nicht im Session State gefunden.")

def display_metadata(metadata: Union[Dict[str, Any], str]):
    """Zeigt Metadaten sicher an"""
    try:
        # Überprüfe, ob metadata ein String ist
        if isinstance(metadata, str):
            st.warning(f"Unerwartetes Metadaten-Format: {metadata}")
            return
            
        if not isinstance(metadata, dict):
            st.warning(f"Ungültiges Metadaten-Format: {type(metadata)}")
            return

        with st.expander("🔍 Details", expanded=False):
            # Intent und Konfidenz
            col1, col2 = st.columns(2)
            with col1:
                intent_info = metadata.get('intent', {})
                if isinstance(intent_info, dict):
                    st.markdown(f"**Intent:** {intent_info.get('intent', 'unknown')}")
                    if metadata.get('subintents'):
                        st.markdown("**Sub-Intents:**")
                        for subintent in metadata['subintents']:
                            st.markdown(f"- {subintent}")
            with col2:
                if isinstance(intent_info, dict):
                    confidence = intent_info.get('confidence', 0.0)
                    st.markdown(f"**Konfidenz:** {confidence:.2f}")

            # Multi-Intent Info
            if metadata.get('multi_intent'):
                st.info("Multiple Intents erkannt")

            # Follow-up Info
            if metadata.get('is_follow_up'):
                st.info("Follow-up Frage erkannt")

            # Erkannte Themen
            if metadata.get('key_topics'):
                st.markdown("**Erkannte Themen:**")
                st.write(", ".join(metadata['key_topics']))

            # Qualitätsmetriken
            if 'quality_metrics' in metadata:
                metrics = metadata['quality_metrics'].get('metrics', {})
                if metrics:
                    st.markdown("### Qualitätsmetriken")
                    metric_data = {
                        'Metrik': ['Relevanz', 'Vollständigkeit', 'Kohärenz', 'Genauigkeit'],
                        'Wert': [
                            f"{metrics.get('relevance_score', 0.0):.2f}",
                            f"{metrics.get('completeness_score', 0.0):.2f}",
                            f"{metrics.get('coherence_score', 0.0):.2f}",
                            f"{metrics.get('accuracy_score', 0.0):.2f}"
                        ]
                    }
                    st.table(metric_data)

            # Error Info
            if 'error' in metadata:
                st.error(metadata['error'])

    except Exception as e:
        logger.error(f"Fehler bei Metadaten-Anzeige: {str(e)}")
        st.warning("Metadaten konnten nicht angezeigt werden")
        
def _get_metric_color(value: float) -> str:
    """Bestimmt die Farbe basierend auf dem Metrik-Wert"""
    if value >= 0.8: return "#00C853"  # Grün
    if value >= 0.6: return "#FFB300"  # Orange
    return "#FF3D00"  # Rot

def add_scroll_button():
    """Fügt einen Scroll-nach-unten Button hinzu"""
    # JavaScript und CSS direkt einfügen
    st.components.v1.html(
        """
        <style>
        .scroll-button {
            position: fixed;
            bottom: 20px;
            right: 20px;
            background-color: rgba(6, 182, 212, 0.8);
            color: white;
            border: none;
            border-radius: 50%;
            width: 40px;
            height: 40px;
            font-size: 24px;
            display: flex;
            align-items: center;
            justify-content: center;
            cursor: pointer;
            transition: opacity 0.3s;
            box-shadow: 0 2px 5px rgba(0,0,0,0.2);
            z-index: 1000; /* Stelle sicher, dass der Button über anderen Elementen liegt */
        }
        .scroll-button:hover {
            opacity: 1;
            background-color: rgba(6, 182, 212, 1);
        }
        </style>
        <script>
        // Erstelle den Button, wenn das Dokument bereit ist
        document.addEventListener('DOMContentLoaded', function() {
            const scrollButton = document.createElement('div');
            scrollButton.classList.add('scroll-button');
            scrollButton.innerHTML = '<i class="fas fa-arrow-down"></i>';
            document.body.appendChild(scrollButton);

            scrollButton.addEventListener('click', () => {
                window.scrollTo(0, document.body.scrollHeight);
            });
        });
        </script>
        <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css">
        """,
        height=0,  # Höhe auf 0 setzen, da wir nur das Skript und CSS einfügen wollen
    )

def add_font_size_control():
    """Fügt eine Schriftgrößen-Kontrolle hinzu"""
    st.sidebar.markdown("### Textgröße")
    font_size = st.sidebar.slider("Wählen Sie die Textgröße", 12, 24, 16)
    st.markdown(
        f"""
        <style>
        .stChatMessage p {{
            font-size: {font_size}px !important;
            line-height: 1.5;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

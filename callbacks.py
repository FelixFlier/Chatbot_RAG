# callbacks.py
import streamlit as st
from typing import Optional

class ProcessCallback:
    """
    Callback-Klasse für die Verarbeitung von Statusupdates
    """
    def __init__(self, status_container):
        self.status_container = status_container
        self.current_step = 0
        self.total_steps = 5  # Anzahl der Verarbeitungsschritte
    
    def update(self, step: str, progress: float = None):
        """
        Aktualisiert den Verarbeitungsstatus
        
        Args:
            step: Beschreibung des aktuellen Schritts
            progress: Optionaler Fortschritt (0-1)
        """
        self.current_step += 1
        if progress is None:
            progress = self.current_step / self.total_steps
        
        try:
            with self.status_container:
                st.progress(progress)
                st.write(step)
        except Exception as e:
            print(f"Fehler beim Status-Update: {str(e)}")

    def reset(self):
        """Setzt den Callback zurück"""
        self.current_step = 0
        try:
            self.status_container.empty()
        except Exception as e:
            print(f"Fehler beim Zurücksetzen des Status: {str(e)}")
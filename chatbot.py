# chatbot_new.py
import os
import time
import pickle
from typing import Dict, List, Any, Optional
import numpy as np
from pathlib import Path
import gdown
from datetime import datetime
from langchain_google_genai import GoogleGenerativeAI
from langchain.memory import ConversationBufferWindowMemory
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from callbacks import ProcessCallback
import streamlit as st
from model_classes import (
    IntelligentIntentRecognizer, 
    DynamicConceptAnalyzer,
    EnhancedThematicTracker, 
    EnhancedHybridSearcher,
    ResponseGenerator
)
from dotenv import load_dotenv
import logging
from shared_models import SharedModels
from optimized_store import OptimizedVectorStore

# Logging Setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(Path(__file__).parent / 'chatbot.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

# Constants
BASE_PATH = Path(__file__).resolve().parent
VECTOR_STORE_PATH = BASE_PATH / "vector_stores"
VECTOR_STORE_FILE = VECTOR_STORE_PATH / "vectorstore.pkl"
BM25_STORE_FILE = VECTOR_STORE_PATH / "bm25.pkl"

CACHE_SIZE = 1000
MIN_SIMILARITY = 0.6
MAX_RESULTS = 5

class InitializationError(Exception):
    """Fehler bei der Initialisierung des Systems"""
    pass

class ChatbotError(Exception):
    """Basis-Fehlerklasse für Chatbot-bezogene Fehler"""
    pass

class APIError(ChatbotError):
    """Fehler bei API-Aufrufen"""
    pass

class StateError(ChatbotError):
    """Fehler im Session State"""
    pass

class IntelligentRAGChatbot:
    def __init__(self, vectorstore_file: Path, bm25_file: Path, api_key: Optional[str] = None):
        """
        Initialisiert den RAG Chatbot mit optimierter Ladestrategie
        """
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        
        try:
            # Initialisiere geteilte Modelle
            self.shared_models = SharedModels()
            self.embeddings_model = self.shared_models.embeddings_model
            
            # Performance Metrics
            self.performance_metrics = []
            self.search_metrics = []
            
            # Vector Store und BM25 Initialisierung
            self._initialize_stores(vectorstore_file, bm25_file)

            # API Key setzen
            if api_key:
                os.environ["GOOGLE_API_KEY"] = api_key

            # Komponenten Initialisierung
            self._initialize_components(api_key)
            
            # Speichere letzte Antwort für Follow-ups
            self.last_response = None
            
        except Exception as e:
            raise InitializationError(f"Fehler bei der Initialisierung: {str(e)}")

    def _initialize_stores(self, vectorstore_file: Path, bm25_file: Path):
        """Initialisiert Vector Store und BM25"""
        try:
            # Überprüfe Dateien
            if not vectorstore_file.exists():
                raise InitializationError(f"Vector Store Datei nicht gefunden: {vectorstore_file}")
            if not bm25_file.exists():
                raise InitializationError(f"BM25 Datei nicht gefunden: {bm25_file}")

            # Optimierte Vector Store Initialisierung
            self.vectorstore = OptimizedVectorStore(vectorstore_file)
            
            # BM25 laden
            with bm25_file.open('rb') as f:
                self.bm25 = pickle.load(f)
                
        except Exception as e:
            raise InitializationError(f"Fehler bei Store-Initialisierung: {str(e)}")

    def _initialize_components(self, api_key: Optional[str] = None):
        try:
            # LLM Setup
            self.llm = self._initialize_llm(api_key)
            
            # Memory
            self.memory = ConversationBufferWindowMemory(
                k=4,
                memory_key="chat_history",
                input_key="question",
                return_messages=True
            )
            
            # Optimierte Komponenten
            self._initialize_optimized_components()
            
            # Speichere letzte Antwort für Follow-ups
            self.last_response = None
                
        except Exception as e:
            raise InitializationError(f"Fehler bei Komponenten-Initialisierung: {str(e)}")

    def _initialize_optimized_components(self):
        try:
            # Thematic Tracker
            self.thematic_tracker = EnhancedThematicTracker(self.embeddings_model)
            
            # Hybrid Searcher
            self.searcher = EnhancedHybridSearcher(
                vectorstore=self.vectorstore,
                bm25=self.bm25,
                thematic_tracker=self.thematic_tracker,
                cache_size=CACHE_SIZE,
                min_similarity=MIN_SIMILARITY
            )
            
            # Intent Recognizer und Response Generator initialisieren
            self.intention_recognizer = IntelligentIntentRecognizer(self.vectorstore, self.llm)
            self.concept_analyzer = DynamicConceptAnalyzer(self.vectorstore)  # Diese Zeile fehlte
            self.response_generator = ResponseGenerator(self.llm, self.embeddings_model)
                
        except Exception as e:
            raise InitializationError(f"Fehler bei Optimierungs-Initialisierung: {str(e)}")
    def _initialize_llm(self, api_key: Optional[str] = None):
        """Initialisiert das LLM"""
        try:
            api_key = api_key or os.getenv("GOOGLE_API_KEY")
            if not api_key:
                raise InitializationError("Google API Key nicht gefunden")
                
            return GoogleGenerativeAI(
                model="gemini-pro",
                temperature=0.4,
                top_p=0.95,
                max_output_tokens=700
            )
        except Exception as e:
            raise InitializationError(f"LLM-Initialisierungsfehler: {str(e)}")

    def get_response(self, query: str, callback: Optional[ProcessCallback] = None) -> Dict[str, Any]:
        start_time = time.time()
        
        try:
            self.logger.debug(f"Starte Verarbeitung für Query: {query}")
            
            # Intent Analyse mit Kontext
            if callback:
                callback.update("Analysiere Intent...", 0.2)
            
            context = {
                'last_response': self.last_response.get('response', '') if self.last_response else '',
                'active_themes': self.thematic_tracker.get_current_context(),
                'last_query': query
            }
            
            intent_info = self.intention_recognizer.analyze_intent(query, context)
            
            # Für reine Konversations-Intents keine Suche durchführen
            if intent_info['intent'] in {'greeting', 'farewell', 'gratitude', 'acknowledgment'}:
                response_info = self.response_generator.generate_response(
                    query=query,
                    context=[],
                    intent_info=intent_info
                )
                self.last_response = response_info
                return response_info

            # Konzeptanalyse
            if callback:
                callback.update("Analysiere Konzepte...", 0.4)
            concept_info = self.concept_analyzer.analyze_query_concepts(query)
            
            # Theme Tracking
            if callback:
                callback.update("Aktualisiere Kontext...", 0.6)
            self.thematic_tracker.update_themes(
                message=query,
                detected_themes=[c.get('text', '') for c in concept_info.get('entities', [])],
                concepts=concept_info,
                intent_info=intent_info
            )

            # Suche nur durchführen wenn nötig
            if not intent_info.get('skip_search', False):
                if callback:
                    callback.update("Suche relevante Informationen...", 0.8)
                search_results = self.searcher.hybrid_search(
                    query=query,
                    context=self.thematic_tracker.get_current_context(),
                    k=MAX_RESULTS
                )
            else:
                search_results = []

            # Antwort generieren
            if callback:
                callback.update("Generiere Antwort...", 0.9)
            response_info = self.response_generator.generate_response(
                query=query,
                context=search_results,
                intent_info=intent_info
            )
            
            # Memory Update
            self._update_memory(query, response_info['response'])
            self.last_response = response_info

            # Performance Tracking
            self._monitor_total_performance(start_time)
            
            return response_info

        except Exception as e:
            self.logger.error(f"Fehler in get_response: {str(e)}", exc_info=True)
            return self._create_error_response(str(e))

    def _update_memory(self, query: str, response: str):
        """Aktualisiert den Konversationskontext"""
        try:
            self.memory.save_context(
                {"question": query},
                {"answer": response}
            )
        except Exception as e:
            self.logger.warning(f"Memory Update fehlgeschlagen: {str(e)}")

    def _monitor_total_performance(self, start_time: float):
        """Überwacht die Gesamtperformance"""
        try:
            duration = time.time() - start_time
            self.performance_metrics.append({
                'total_duration': duration,
                'timestamp': datetime.now()
            })
            
            if len(self.performance_metrics) > CACHE_SIZE:
                self.performance_metrics = self.performance_metrics[-CACHE_SIZE:]
                
        except Exception as e:
            self.logger.warning(f"Performance Monitoring fehlgeschlagen: {str(e)}")

    def _create_error_response(self, error_message: str) -> Dict[str, Any]:
        """Erstellt eine formatierte Fehlerantwort"""
        return {
            'response': "Es tut mir leid, es ist ein Fehler aufgetreten. Bitte versuchen Sie es erneut.",
            'metadata': {
                'error': error_message,
                'intent': {'intent': 'error', 'confidence': 0.0},
                'quality_metrics': {'metrics': {}},
                'context_coverage': 0.0
            }
        }

    def get_performance_stats(self) -> Dict[str, Any]:
        """Liefert Performance-Statistiken"""
        try:
            if not self.performance_metrics:
                return {}
            
            recent_metrics = self.performance_metrics[-100:]
            
            return {
                'average_response_time': np.mean([m['total_duration'] for m in recent_metrics]),
                'total_requests': len(self.performance_metrics),
                'cache_hit_rate': self.searcher.get_cache_stats() if hasattr(self.searcher, 'get_cache_stats') else None
            }
        except Exception as e:
            self.logger.error(f"Fehler beim Abrufen der Performance-Statistiken: {str(e)}")
            return {}

    def clear_caches(self):
        """Leert alle Caches"""
        try:
            if hasattr(self.searcher, 'result_cache'):
                self.searcher.result_cache.clear()
            if hasattr(self.searcher, 'embedding_cache'):
                self.searcher.embedding_cache.clear()
            self.logger.info("Caches erfolgreich geleert")
        except Exception as e:
            self.logger.error(f"Fehler beim Leeren der Caches: {str(e)}")

import os
import pickle
from typing import Dict, List, Any, Optional, Callable
import numpy as np
from pathlib import Path
from langchain_google_genai import GoogleGenerativeAI
from langchain.memory import ConversationBufferWindowMemory
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from callbacks import ProcessCallback
from greeting_handler import GreetingRecognizer, GreetingResponder
from model_classes import (
    IntelligentIntentRecognizer, 
    DynamicConceptAnalyzer,
    EnhancedThematicTracker, 
    EnhancedHybridSearcher,
    ResponseGenerator
)
from dotenv import load_dotenv
load_dotenv()
# Constants
BASE_PATH = Path(__file__).parent
VECTOR_STORE_PATH = str(BASE_PATH / "vector_stores")

class InitializationError(Exception):
    """Fehler bei der Initialisierung des Chatbots"""
    pass

class ProcessingError(Exception):
    """Fehler bei der Verarbeitung einer Anfrage"""
    pass

class IntelligentRAGChatbot:
    def __init__(self, vectorstore_path: str, bm25_path: str, api_key: Optional[str] = None):
        """
        Initialisiert den RAG Chatbot
        
        Args:
            vectorstore_path: Pfad zur Vector Store Datei
            bm25_path: Pfad zur BM25 Datei
            api_key: Optional - Google API Key
        """
        try:
            # Verbesserte Pfadhandhabung
            self.base_path = Path(os.path.dirname(os.path.abspath(__file__)))
            self.vector_store_path = self.base_path / "vector_stores"
            
            # Erstelle Vector Store Verzeichnis falls nicht vorhanden
            self.vector_store_path.mkdir(parents=True, exist_ok=True)
            
            self.greeting_recognizer = GreetingRecognizer()
            self.greeting_responder = GreetingResponder()
            
            # Initialisiere Embeddings mit Error Handling
            try:
                self.embeddings = HuggingFaceEmbeddings(
                    model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
                )
            except Exception as e:
                raise InitializationError(f"Fehler bei der Initialisierung der Embeddings: {str(e)}")
            
            vectorstore_full_path = os.path.join(VECTOR_STORE_PATH, vectorstore_path)
            bm25_full_path = os.path.join(VECTOR_STORE_PATH, bm25_path)
            
            with open(vectorstore_full_path, 'rb') as f:
                self.vectorstore = pickle.load(f)
            with open(bm25_full_path, 'rb') as f:
                self.bm25 = pickle.load(f)
            
            # API Key Handling
            if api_key:
                os.environ["GOOGLE_API_KEY"] = api_key
            elif not os.getenv("GOOGLE_API_KEY"):
                raise InitializationError("GOOGLE_API_KEY nicht gefunden in Umgebungsvariablen")
            
            # LLM Initialisierung mit Error Handling
            try:
                self.llm = GoogleGenerativeAI(
                    model="gemini-pro",
                    temperature=0.4,
                    top_p=0.95,
                    max_output_tokens=700
                )
            except Exception as e:
                raise InitializationError(f"Fehler bei der LLM Initialisierung: {str(e)}")
            
            # Memory Initialisierung
            self.memory = ConversationBufferWindowMemory(
                k=4,
                memory_key="chat_history",
                input_key="question",
                return_messages=True  # Aktiviere Message Format
            )
            
            # Komponenten Initialisierung mit Error Handling
            try:
                self.thematic_tracker = EnhancedThematicTracker(self.embeddings)
                self.searcher = EnhancedHybridSearcher(
                    self.vectorstore,
                    self.bm25,
                    self.thematic_tracker
                )
                self.intention_recognizer = IntelligentIntentRecognizer(self.vectorstore)
                self.concept_analyzer = DynamicConceptAnalyzer(self.vectorstore)
                self.response_generator = ResponseGenerator(
                    self.llm,
                    self.embeddings
                )
            except Exception as e:
                raise InitializationError(f"Fehler bei der Komponenten-Initialisierung: {str(e)}")
            
        except Exception as e:
            raise InitializationError(f"Fehler bei der Initialisierung: {str(e)}")

    def get_relevant_context(self, query: str) -> str:
        """
        Holt relevanten Kontext aus dem Vector Store mit verbessertem Error Handling
        """
        try:
            if not query or not isinstance(query, str):
                raise ValueError("Ungültige Anfrage")
                
            results = self.searcher.hybrid_search(
                query,
                self.thematic_tracker.get_current_context()
            )
            
            if not results:
                return ""
                
            # Validiere Ergebnisse
            if not isinstance(results, (list, tuple)):
                raise TypeError("Ungültiges Ergebnisformat vom Searcher")
                
            return " ".join(str(r) for r in results)
            
        except Exception as e:
            print(f"Fehler bei der Kontextsuche: {str(e)}")
            return ""

    
    def get_response(self, query: str, callback: Optional[ProcessCallback] = None) -> Dict[str, Any]:
        try:
            # Prüfe zuerst auf Begrüßung
            greeting_info = self.greeting_recognizer.analyze_greeting(query)
            if greeting_info['is_greeting']:
                response = self.greeting_responder.get_response(greeting_info['type'])
                return {
                    'answer': response,
                    'metadata': {
                        'intent': {'intent': 'greeting', 'confidence': greeting_info['confidence']},
                        'quality_metrics': {
                            'metrics': {
                                'relevance_score': 1.0,
                                'completeness_score': 1.0,
                                'coherence_score': 1.0,
                                'accuracy_score': 1.0
                            }
                        },
                        'context_coverage': 1.0,
                        'failed_quality_check': False,
                        'failed_metrics': []
                    }
                }

            # Normale Verarbeitung
            if callback:
                callback.update("Analysiere Intent...", 0.2)
            
            intent_info = self.intention_recognizer.analyze_intent(
                query,
                self.thematic_tracker.get_current_context()
            )
            
            if callback:
                callback.update("Analysiere Konzepte...", 0.4)
            
            concept_info = self.concept_analyzer.analyze_query_concepts(query)
            
            if callback:
                callback.update("Aktualisiere thematischen Kontext...", 0.6)
                
            theme_updates = self.thematic_tracker.update_themes(
                message=query,
                detected_themes=[concept['text'] for concept in 
                            concept_info.get('query_concepts', {}).get('entities', [])],
                concepts=concept_info.get('combined_concepts', {}),
                intent_info=intent_info
            )
            
            if callback:
                callback.update("Suche relevante Informationen...", 0.7)
            
            search_results = self.searcher.hybrid_search(
                query,
                self.thematic_tracker.get_current_context()
            )
            
            if not search_results:
                if callback:
                    callback.update("Keine relevanten Informationen gefunden", 1.0)
                return {
                    'answer': "Ich konnte leider keine relevanten Informationen finden.",
                    'metadata': {
                        'intent': intent_info,
                        'concepts': concept_info,
                        'theme_updates': theme_updates,
                        'quality_metrics': {'metrics': {}},
                        'context_coverage': 0.0
                    }
                }

            if callback:
                callback.update("Generiere Antwort...", 0.9)
            
            response_info = self.response_generator.generate_response(
                query,
                search_results,
                intent_info
            )
            
            # Hole die Antwort und Qualitätsmetriken
            answer_text = response_info.get('answer', '')
            quality_check = response_info.get('metadata', {}).get('quality_metrics', {})
            failed_metrics = response_info.get('metadata', {}).get('failed_metrics', [])
            
            # Aktualisiere Gedächtnis
            self.memory.save_context(
                {"question": query},
                {"answer": answer_text}
            )

            if callback:
                callback.update("Fertig!", 1.0)

            return {
                'answer': answer_text,
                'metadata': {
                    'intent': intent_info,
                    'concepts': concept_info,
                    'theme_updates': theme_updates,
                    'quality_metrics': quality_check,
                    'context_coverage': response_info.get('metadata', {}).get('context_coverage', 0.0),
                    'failed_quality_check': bool(failed_metrics),
                    'failed_metrics': failed_metrics
                }
            }
                
        except Exception as e:
            if callback:
                callback.update(f"Fehler: {str(e)}", 1.0)
            return {
                'answer': "Es tut mir leid, aber ich konnte keine angemessene Antwort generieren. "
                        "Können Sie Ihre Frage bitte anders formulieren?",
                'error': str(e),
                'metadata': {
                    'quality_metrics': {'metrics': {}},
                    'context_coverage': 0.0,
                    'failed_quality_check': True,
                    'failed_metrics': ['processing_error']
                }
            }

    def _process_query(self, query: str) -> Dict[str, Any]:
        """
        Interne Methode zur Verarbeitung der Anfrage
        """
        # 1. Intentionserkennung
        intent_info = self.intention_recognizer.analyze_intent(
            query,
            self.thematic_tracker.get_current_context()
        )
        
        # 2. Konzeptanalyse
        concept_info = self.concept_analyzer.analyze_query_concepts(query)
        
        # 3. Thematisches Tracking
        theme_updates = self.thematic_tracker.update_themes(
            message=query,
            detected_themes=[concept['text'] for concept in 
                        concept_info.get('query_concepts', {}).get('entities', [])],
            concepts=concept_info.get('combined_concepts', {}),
            intent_info=intent_info
        )
        
        # 4. Kontextsuche
        search_results = self.searcher.hybrid_search(
            query,
            self.thematic_tracker.get_current_context()
        )
        
        if not search_results:
            return {
                'answer': "Ich konnte leider keine relevanten Informationen finden.",
                'metadata': {
                    'intent': intent_info,
                    'concepts': concept_info,
                    'theme_updates': theme_updates
                },
                'success': True
            }
        
        # 5. Antwortgenerierung
        response_info = self.response_generator.generate_response(
            query,
            search_results,
            intent_info
        )
        
        # 6. Gedächtnis aktualisieren
        answer_text = response_info['answer']
        self.memory.save_context(
            {"question": query},
            {"answer": answer_text}
        )
        
        # 7. Erfolgreiche Antwort
        return {
            'answer': answer_text,
            'metadata': {
                'intent': intent_info,
                'concepts': concept_info,
                'theme_updates': theme_updates,
                'quality_metrics': response_info.get('quality_metrics'),
                'context_coverage': response_info.get('context_coverage')
            },
            'success': True
        }

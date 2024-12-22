# model_classes.py
from dataclasses import dataclass, field
from typing import Dict, List, Set, Optional, Tuple, Any, Callable
from datetime import datetime, timedelta
import spacy
import numpy as np
from collections import defaultdict
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from rank_bm25 import BM25Okapi
import re  # Am Anfang der Datei
from concurrent.futures import ThreadPoolExecutor, TimeoutError
import networkx as nx
import numpy as np
from typing import List, Dict, Any, Optional
from collections import defaultdict
import time
from scipy.spatial.distance import cosine
from langchain.schema import Document
from scipy.sparse import csr_matrix, vstack
from sklearn.preprocessing import normalize
from langchain.schema import Document
from sklearn.metrics.pairwise import cosine_similarity
import concurrent.futures
import heapq
from scipy.sparse import csr_matrix, vstack
from langchain.docstore.document import Document
from functools import lru_cache
import traceback
from datetime import datetime
from langchain.chains import LLMChain
import logging
from pathlib import Path
from shared_models import SharedModels

# Setup logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(Path(__file__).parent / 'model_classes.log'),
        logging.StreamHandler()
    ]
)

# Create logger instance
logger = logging.getLogger(__name__)


from typing import Dict, Any, List, Optional
from datetime import datetime
import numpy as np
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

class IntelligentIntentRecognizer:
    """Enhanced intent recognition with LLM integration"""
    
    def __init__(self, vectorstore, llm):
        self.vectorstore = vectorstore
        self.llm = llm
        self.intent_history = []
        
        # Core intents with response strategies
        self.intent_categories = {
            # Conversation Control Intents
            "greeting": {
                "description": "User greets the system",
                "examples": ["hallo", "hi", "guten tag", "guten morgen", "moin", "servus"],
                "response_type": "greeting"
            },
            "farewell": {
                "description": "User says goodbye",
                "examples": ["tschüss", "auf wiedersehen", "bye", "ciao", "bis später"],
                "response_type": "farewell"
            },
            "gratitude": {
                "description": "User expresses thanks",
                "examples": ["danke", "vielen dank", "super danke", "perfekt danke", "okay danke"],
                "response_type": "gratitude"
            },
            "acknowledgment": {
                "description": "User acknowledges information",
                "examples": ["okay", "alles klar", "verstanden", "ich verstehe", "gut"],
                "response_type": "acknowledgment"
            },
            "positive_feedback": {
                "description": "User gives positive feedback",
                "examples": ["das ist hilfreich", "super", "toll", "sehr gut", "perfekt"],
                "response_type": "positive_feedback"
            },
            "negative_feedback": {
                "description": "User gives negative feedback",
                "examples": ["das hilft nicht", "nicht hilfreich", "verstehe ich nicht", "zu kompliziert"],
                "response_type": "negative_feedback"
            },
            "follow_up": {
                "description": "User asks for clarification or additional information",
                "examples": ["kannst du das genauer erklären", "was bedeutet das", "wie meinst du das", "und weiter"],
                "response_type": "follow_up"
            },
            "multi_intent": {
                "description": "User combines multiple intents",
                "examples": ["danke, und kannst du noch", "okay verstanden, aber was bedeutet"],
                "response_type": "multi_intent"
            },
            # Information Intents
            "information": {
                "description": "User seeks factual information",
                "examples": ["was ist", "erkläre", "beschreibe", "bedeutet"],
                "response_type": "information"
            },
            "process": {
                "description": "User wants to understand a process or procedure",
                "examples": ["wie läuft", "workflow", "prozess", "ablauf", "schritte"],
                "response_type": "process"
            },
            "comparison": {
                "description": "User wants to compare concepts",
                "examples": ["unterschied", "vergleich", "versus", "im gegensatz zu"],
                "response_type": "comparison"
            },
            "clarification": {
                "description": "User needs clarification or has follow-up",
                "examples": ["kannst du das genauer", "was meinst du", "verstehe nicht"],
                "response_type": "clarification"
            },
            "application": {
                "description": "User wants practical application information",
                "examples": ["beispiel", "anwendung", "praxis", "konkret"],
                "response_type": "application"
            },
            "meta": {
                "description": "Questions about the conversation itself",
                "examples": ["kannst du", "bist du in der lage", "verstehst du"],
                "response_type": "meta"
            }
        }

        # Initialize LLM chain for intent recognition
        self.intent_prompt = PromptTemplate(
            input_variables=["query", "context", "intent_categories", "last_response"],
            template="""Analysiere die folgende Benutzeranfrage unter Berücksichtigung des Kontexts. 
            Identifiziere alle Intents und prüfe auf Follow-ups oder mehrfache Absichten.

            Benutzeranfrage: {query}
            
            Vorherige Antwort: {last_response}
            Kontext: {context}
            
            Mögliche Intent-Kategorien:
            {intent_categories}
            
            Gib deine Analyse im folgenden Format zurück:
            Intent: [Hauptintent]
            Subintents: [Liste weiterer erkannter Intents, falls vorhanden]
            Is_Follow_Up: [true/false]
            Follow_Up_Reference: [Bezug zur vorherigen Antwort, falls vorhanden]
            Confidence: [0-1]
            Requires_Context: [true/false]
            Key_Topics: [Wichtige erkannte Themen]
            Multi_Intent: [true/false]
            Intent_Sequence: [Reihenfolge der zu verarbeitenden Intents]
            """)
        
        self.intent_chain = LLMChain(llm=self.llm, prompt=self.intent_prompt)

    # In model_classes.py - IntelligentIntentRecognizer Klasse

    def analyze_intent(self, query: str, context: Optional[Dict] = None) -> Dict[str, Any]:
        """Analyzes the query for intent"""
        try:
            # Check for conversation intents first
            conversation_intent = self._check_conversation_intent(query)
            if conversation_intent:
                return conversation_intent

            # Continue with normal intent analysis
            context = context or {}
            llm_response = self.intent_chain.invoke({
                "query": query,
                "context": self._prepare_context_string(context),
                "intent_categories": self._format_intent_categories(),
                "last_response": context.get('last_response', '')
            })
            
            intent_info = self._parse_llm_response(llm_response['text'])
            pattern_info = self._pattern_analysis(query)
            final_intent = self._combine_analyses(intent_info, pattern_info)
            
            return final_intent

        except Exception as e:
            logger.error(f"Error in intent analysis: {str(e)}")
            return {
                'intent': 'information',
                'confidence': 0.5,
                'response_type': 'information'
            }

    def _check_conversation_intent(self, query: str) -> Optional[Dict[str, Any]]:
        """Checks for conversation control intents"""
        query_lower = query.lower().strip()
        
        intent_patterns = {
            'greeting': ['hallo', 'hi', 'guten tag', 'guten morgen', 'guten abend', 'moin'],
            'farewell': ['tschüss', 'auf wiedersehen', 'bye', 'ciao', 'bis später'],
            'gratitude': ['danke', 'vielen dank', 'super danke', 'perfekt danke'],
            'acknowledgment': ['okay', 'alles klar', 'verstanden', 'gut']
        }

        for intent_type, patterns in intent_patterns.items():
            if any(pattern in query_lower for pattern in patterns):
                return {
                    'intent': intent_type,
                    'confidence': 1.0,
                    'response_type': intent_type  # Konsistent mit dem Intent-Typ
                }
        
        return None
    def _format_intent_categories(self) -> str:
        """Formats intent categories for LLM prompt"""
        formatted_categories = []
        for intent, details in self.intent_categories.items():
            formatted_categories.append(
                f"- {intent}: {details['description']}\n  Beispiele: {', '.join(details['examples'])}"
            )
        return "\n".join(formatted_categories)

    def _prepare_context_string(self, context: Optional[Dict]) -> str:
        """Prepares context information for the LLM prompt"""
        if not context:
            return "Kein vorheriger Kontext verfügbar."
            
        context_elements = []
        
        if 'last_query' in context:
            context_elements.append(f"Letzte Anfrage: {context['last_query']}")
            
        if 'active_themes' in context:
            themes = context['active_themes']
            context_elements.append(f"Aktive Themen: {', '.join(themes)}")
            
        if 'last_intent' in context:
            context_elements.append(f"Letzter Intent: {context['last_intent']}")
            
        return " | ".join(context_elements)

    def _parse_llm_response(self, response: str) -> Dict[str, Any]:
        """Parses the LLM response into a structured format"""
        lines = response.strip().split('\n')
        parsed = {
            'intent': 'information',
            'subintents': [],
            'is_follow_up': False,
            'follow_up_reference': '',
            'confidence': 0.7,
            'requires_context': False,
            'key_topics': [],
            'multi_intent': False,
            'intent_sequence': []
        }
        
        for line in lines:
            if ':' not in line:
                continue
                
            key, value = line.split(':', 1)
            key = key.strip().lower()
            value = value.strip()
            
            if key == 'intent':
                parsed['intent'] = value.lower()
            elif key == 'subintents':
                parsed['subintents'] = [s.strip() for s in value.strip('[]').split(',') if s.strip()]
            elif key == 'is_follow_up':
                parsed['is_follow_up'] = value.lower() == 'true'
            elif key == 'follow_up_reference':
                parsed['follow_up_reference'] = value
            elif key == 'confidence':
                try:
                    parsed['confidence'] = float(value)
                except ValueError:
                    pass
            elif key == 'requires_context':
                parsed['requires_context'] = value.lower() == 'true'
            elif key == 'key_topics':
                parsed['key_topics'] = [t.strip() for t in value.strip('[]').split(',') if t.strip()]
            elif key == 'multi_intent':
                parsed['multi_intent'] = value.lower() == 'true'
            elif key == 'intent_sequence':
                parsed['intent_sequence'] = [i.strip() for i in value.strip('[]').split(',') if i.strip()]
                
        return parsed

    def _pattern_analysis(self, query: str) -> Dict[str, Any]:
        """Performs basic pattern matching as backup"""
        query_lower = query.lower()
        
        # Find matching patterns
        matched_intents = {}
        for intent, details in self.intent_categories.items():
            matches = sum(1 for pattern in details['examples'] 
                        if pattern in query_lower)
            if matches:
                matched_intents[intent] = matches
                
        if not matched_intents:
            return {
                'intent': 'information',
                'confidence': 0.5,
                'method': 'pattern'
            }
            
        # Select best match
        best_intent = max(matched_intents.items(), key=lambda x: x[1])[0]
        confidence = min(matched_intents[best_intent] / 3, 1.0)  # Normalize confidence
        
        return {
            'intent': best_intent,
            'confidence': confidence,
            'method': 'pattern'
        }

    def _combine_analyses(
        self,
        llm_analysis: Dict[str, Any],
        pattern_analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Combines LLM and pattern matching results"""
        # Use LLM intent if confidence is high enough
        if llm_analysis['confidence'] >= 0.7:
            final_intent = llm_analysis['intent']
            confidence = llm_analysis['confidence']
        # Fall back to pattern matching if LLM confidence is low
        elif pattern_analysis['confidence'] >= 0.5:
            final_intent = pattern_analysis['intent']
            confidence = pattern_analysis['confidence']
        # Use default if both methods have low confidence
        else:
            final_intent = 'information'
            confidence = 0.5

        return {
            'intent': final_intent,
            'subintents': llm_analysis.get('subintents', []),
            'confidence': confidence,
            'key_topics': llm_analysis.get('key_topics', []),
            'requires_context': llm_analysis.get('requires_context', False),
            'response_type': self.intent_categories[final_intent]['response_type']
        }

    def _update_intent_history(self, intent_info: Dict[str, Any], query: str):
        """Updates the intent history"""
        self.intent_history.append({
            'intent': intent_info['intent'],
            'query': query,
            'timestamp': datetime.now(),
            'confidence': intent_info['confidence']
        })
        
        # Keep history manageable
        if len(self.intent_history) > 10:
            self.intent_history.pop(0)

    def _create_fallback_intent(self) -> Dict[str, Any]:
        """Creates a safe fallback intent"""
        return {
            'intent': 'information',
            'subintents': [],
            'confidence': 0.5,
            'key_topics': [],
            'requires_context': False,
            'response_type': 'information'
        }

    def get_intent_history(self) -> List[Dict[str, Any]]:
        """Returns the intent history"""
        return self.intent_history.copy()

class DynamicConceptAnalyzer:
    """Vereinfachte Konzeptanalyse"""
    
    def __init__(self, vectorstore):
        self.vectorstore = vectorstore
        self.shared_models = SharedModels()  # Verwende geteilte Modelle
        self.concept_cache = {}
        self.entity_cache = {}
    
    def analyze_query_concepts(self, query: str, context_window: int = 3) -> Dict[str, Any]:
        if query in self.concept_cache:
            return self.concept_cache[query]
        
        doc = self.shared_models.spacy_model(query)
        
        concepts = {
            'query_concepts': {
                'main_topic': self._identify_main_topic(doc),
                'entities': self._extract_entities(doc)
            }
        }
        
        self.concept_cache[query] = concepts
        return concepts

    def _identify_main_topic(self, doc) -> Optional[str]:
        subjects = [token for token in doc if "subj" in token.dep_]
        return subjects[0].text if subjects else None
    
    def _extract_entities(self, doc) -> List[Dict[str, str]]:
        cache_key = doc.text[:100]  # Ersten 100 Zeichen als Cache-Key
        if cache_key in self.entity_cache:
            return self.entity_cache[cache_key]
            
        entities = [
            {
                'text': ent.text,
                'label': ent.label_,
                'description': spacy.explain(ent.label_)
            }
            for ent in doc.ents
        ]
        self.entity_cache[cache_key] = entities
        if len(self.entity_cache) > 1000:  # Cache-Größe begrenzen
            self.entity_cache.pop(next(iter(self.entity_cache)))
        return entities
    
    def _extract_noun_phrases(self, doc) -> List[str]:
        """Extrahiert Nominalphrasen"""
        return [chunk.text for chunk in doc.noun_chunks]
    
    def _extract_key_terms(self, text: str) -> List[Tuple[str, float]]:
        """Extrahiert Schlüsselbegriffe"""
        return self.keyword_model.extract_keywords(
            text,
            keyphrase_ngram_range=(1, 3),
            stop_words='german',
            top_n=5
        )
    
    def _extract_relations(self, doc) -> List[Dict[str, str]]:
        """Extrahiert semantische Beziehungen zwischen Konzepten"""
        relations = []
        for token in doc:
            if token.dep_ in ['ROOT', 'nsubj', 'dobj', 'pobj']:
                relation = {
                    'source': token.head.text,
                    'relation_type': token.dep_,
                    'target': token.text,
                    'sentence': token.sent.text
                }
                # Füge zusätzliche Kontextinformationen hinzu
                if token.head.ent_type_ and token.ent_type_:
                    relation.update({
                        'source_type': token.head.ent_type_,
                        'target_type': token.ent_type_
                    })
                relations.append(relation)
        return relations

    def _combine_concepts(self, query_concepts: Dict, 
                         doc_concepts: Dict) -> Dict[str, Any]:
        """Kombiniert und gewichtet Konzepte aus Query und Dokumenten"""
        combined = {
            'primary_concepts': set(),
            'secondary_concepts': set(),
            'relations': [],
            'confidence_scores': {}
        }
        
        # Primäre Konzepte aus der Query
        if query_concepts.get('main_topic'):
            combined['primary_concepts'].add(query_concepts['main_topic'])
        
        # Füge Entitäten hinzu
        for ent in query_concepts.get('entities', []):
            combined['primary_concepts'].add(ent['text'])
        
        # Dokumentenkonzepte als sekundäre Konzepte
        for concept_list in doc_concepts.values():
            if isinstance(concept_list, list):
                for concept in concept_list:
                    if isinstance(concept, dict):
                        combined['secondary_concepts'].add(concept.get('text', ''))
                    else:
                        combined['secondary_concepts'].add(str(concept))
        
        # Berechne Konfidenzwerte
        for concept in combined['primary_concepts']:
            confidence = self._calculate_concept_confidence(
                concept, query_concepts, doc_concepts
            )
            combined['confidence_scores'][concept] = confidence
        
        return combined

    def _calculate_concept_confidence(self, concept: str,
                                    query_concepts: Dict,
                                    doc_concepts: Dict) -> float:
        """Berechnet Konfidenzwert für ein Konzept"""
        confidence = 0.0
        
        # Erhöhe Konfidenz wenn Konzept in Query vorkommt
        if concept == query_concepts.get('main_topic'):
            confidence += 0.5
        
        # Prüfe Vorkommen in Dokumenten
        for doc_concept_list in doc_concepts.values():
            if isinstance(doc_concept_list, list):
                for doc_concept in doc_concept_list:
                    if isinstance(doc_concept, dict):
                        if concept == doc_concept.get('text'):
                            confidence += 0.3
                    elif concept == str(doc_concept):
                        confidence += 0.3
        
        return min(confidence, 1.0)

from sklearn.preprocessing import normalize
from scipy.sparse import csr_matrix, vstack
from langchain.docstore.document import Document
from collections import defaultdict
from datetime import datetime
from typing import Any, Dict, List, Optional
import concurrent.futures
import numpy as np
import networkx as nx
from sklearn.preprocessing import normalize
from scipy.sparse import csr_matrix, vstack
from langchain.docstore.document import Document
import spacy
import traceback

class EnhancedHybridSearcher:
    """Optimierte Hybrid-Suche für RAG Chatbots mit verbesserten Suchstrategien"""

    def __init__(
        self,
        vectorstore,
        bm25,
        thematic_tracker,
        cache_size: int = 1000,
        min_similarity: float = 0.2,
        nlp_model: str = "de_core_news_sm"
    ):

        # Schwellenwerte für die Filterung und adaptive Suche
        self.quality_thresholds = {
            "stage1": {
                "semantic": 0.3,    # Gesenkt von 0.4 (da avg_score meist unter 0.6)
                "keyword": 0.15     # Gesenkt von 0.1 (da sehr niedrige Scores)
            },
            "stage2": {
                "graph": 0.05,      # Neu angepasst basierend auf avg_score ≈ 0.036
                "statistical": 0.05  # Neu angepasst für statistische Suche
            }
        }
        """Initialisiert den Searcher mit allen benötigten Komponenten"""
        # Basis-Komponenten
        self.vectorstore = vectorstore
        self.bm25 = bm25
        self.thematic_tracker = thematic_tracker

        # Neue Zeile: Initialisiere documents
        self.documents = self._get_all_documents()

        # NLP-Modell für die Vorverarbeitung
        self.nlp = spacy.load(nlp_model)

        # Konfiguration
        self.cache_size = cache_size
        self.minimum_similarity_threshold = min_similarity

        self.result_cache = {}
        self.cache_ttl = 3600  # 1 Stunde
        self.last_cache_cleanup = datetime.now()
        self.embedding_cache = {}

        # Optimierungs-Strukturen
        self.sparse_matrices = {}
        self.term_frequency = defaultdict(float)
        self.document_frequency = defaultdict(int)
        self.term_to_index = {}

        # Such-Gewichte (angepasst)
        self.search_weights = {
            "semantic": 0.6,    # Erhöht aufgrund guter Performance
            "keyword": 0.1,    # Reduziert aufgrund schlechterer Performance
            "graph": 0.15,       # Reduziert wegen höherer Latenz
            "statistical": 0.1,  # Beibehalten wegen guter Latenz
            "context": 0.05     # Reduziert als Ausgleich
        }

        # Performance-Tracking
        self.performance_metrics = defaultdict(list)

        # Initialisierung der Komponenten
        self._initialize_matrices()
        self.chunk_graph = self._build_chunk_graph()
        
        self.section_index = {}
        self.hierarchy_weights = {
            'level_1': 1.0,  # Hauptthemen
            'level_2': 0.8,  # Unterkapitel
            'level_3': 0.6   # Aufzählungspunkte
        }
        # Kontext-Parameter
        self.context_window = 2  # Anzahl der Nachbar-Chunks
        self.min_context_similarity = 0.3 

    # 1. Hauptsuchfunktion
    def hybrid_search(self, query: str, context: Optional[Dict[str, Any]] = None, k: int = 5) -> List[Dict[str, Any]]:
        """Führt eine mehrstufige hybride Suche durch"""
        try:
            if context is None:
                context = {}

            # Cache-Check
            cache_key = f"{query}_{str({k: v for k, v in context.items() if k in ['sources', 'active_themes']})}"
            if cache_key in self.result_cache:
                return self.result_cache[cache_key]

            # Query-Typ erkennen
            query_type = self._detect_query_type(query)
            print(f"Erkannter Query-Typ: {query_type}")

            # STUFE 1: Schnelle Vorauswahl
            stage1_results = {
                "semantic": self._semantic_search(query, k=20),
                "keyword": self._keyword_search(query, k=min(5 + len(query.split()), 20))
            }
            stage1_quality = self._evaluate_stage_quality(stage1_results, 1)
            stage1_results = self._apply_quality_filter(stage1_results)

            # STUFE 2: Erweiterte Suche falls nötig
            # Nur avg_score als Kriterium verwenden
            if stage1_quality["avg_score"] < 0.4:  # Gesenkt von 0.6 auf 0.4
 
                stage2_results = {
                    "graph": self._graph_search(query, k),
                    "statistical": self._statistical_search(query, k)
                }
                search_results = {**stage1_results, **stage2_results}
                search_results = self._apply_quality_filter(search_results)
            else:
                search_results = stage1_results

            # STUFE 3: Verfeinerung und Kombination
            weights = self._calculate_adaptive_weights(query, context)
            weights = self._adjust_weights_based_on_quality(weights, stage1_quality, query_type)

            # Kombiniere und ranke Ergebnisse
            combined_results = self._combine_all_results(search_results, query, context, weights)
            final_results = combined_results[:k]

            # Cache-Update
            self._update_cache(cache_key, final_results)
            self._update_performance_metrics(query, final_results, datetime.now())

            return final_results

        except Exception as e:
            print(f"Fehler in hybrid_search: {str(e)}")
            print(f"Vollständiger Traceback: {traceback.format_exc()}")
            return []

    # 2. Initialisierungsfunktionen
    def _initialize_matrices(self):
        """Initialisiert Sparse-Matrizen für optimierte Suche und TF-IDF Berechnung"""
        try:
            documents = self._get_all_documents()

            rows, cols, data = [], [], []
            tf_idf_rows, tf_idf_cols, tf_idf_data = [], [], []
            doc_lengths = []

            for doc_id, doc in enumerate(documents):
                if doc and isinstance(doc, Document):
                    # Verbesserte Vorverarbeitung mit spaCy
                    preprocessed_text = self._preprocess_text(doc.page_content)
                    terms = preprocessed_text
                    doc_lengths.append(len(terms))
                    term_counts = defaultdict(int)

                    for term in terms:
                        term_counts[term] += 1
                        self.term_frequency[term] += 1

                    for term, count in term_counts.items():
                        self.document_frequency[term] += 1
                        # Nutzung des term_to_index Mappings
                        if term not in self.term_to_index:
                            self.term_to_index[term] = len(self.term_to_index)
                        term_idx = self.term_to_index[term]

                        rows.append(doc_id)
                        cols.append(term_idx)
                        data.append(count)

                        # TF-IDF Berechnung
                        tf = count / len(terms)
                        idf = np.log(
                            1 + len(documents) / (1 + self.document_frequency[term])
                        )
                        tf_idf_rows.append(doc_id)
                        tf_idf_cols.append(term_idx)
                        tf_idf_data.append(tf * idf)

            # Normalisierung der Dokumentenlängen
            avg_doc_length = np.mean(doc_lengths)
            k1 = 1.5  # BM25 Parameter
            b = 0.75  # BM25 Parameter
            for i in range(len(tf_idf_rows)):
                tf_idf_data[i] *= (k1 + 1) / (
                    tf_idf_data[i]
                    + k1 * (1 - b + b * doc_lengths[tf_idf_rows[i]] / avg_doc_length)
                )

            # Erstellen der Sparse Matrizen
            self.sparse_matrices["term_doc"] = normalize(csr_matrix((data, (rows, cols)), shape=(len(documents), len(self.term_to_index))))
            self.sparse_matrices["tf_idf"] = csr_matrix(
                (tf_idf_data, (tf_idf_rows, tf_idf_cols)), shape=(len(documents), len(self.term_to_index))
            )

        except Exception as e:
            print(f"Fehler bei Matrix-Initialisierung: {str(e)}")

    def _build_chunk_graph(self) -> nx.DiGraph:
        """Erstellt Graph-Struktur für Dokumentbeziehungen"""
        graph = nx.DiGraph()
        try:
            documents = self._get_all_documents()

            # Knoten hinzufügen
            for i, doc in enumerate(documents):
                if doc:
                    embedding = self._get_or_create_embedding(doc.page_content)
                    graph.add_node(
                        i,
                        content=doc.page_content,
                        metadata=getattr(doc, 'metadata', {}),
                        embedding=embedding,
                        quality_metrics=self._calculate_quality_metrics(doc.page_content)
                    )

            # Kanten hinzufügen (verbessert)
            self._add_graph_edges_improved(graph, documents)

            return graph

        except Exception as e:
            print(f"Fehler beim Erstellen des Chunk-Graphen: {str(e)}")
            return graph

    # 3. Suchmethoden
    def _parallel_search(self, query: str, k: int) -> Dict[str, List[Dict]]:
        """Führt verschiedene Suchmethoden parallel aus"""
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = {
                "semantic": executor.submit(self._semantic_search, query, k),
                "keyword": executor.submit(self._keyword_search, query, k),
                "graph": executor.submit(self._graph_search, query, k),
                "statistical": executor.submit(self._statistical_search, query, k),
            }

            results = {}
            for key, future in futures.items():
                try:
                    result = future.result()
                    # Validiere das Ergebnis
                    if not isinstance(result, list):
                        print(f"WARNUNG: {key} search gab kein List-Objekt zurück: {type(result)}")
                        result = []
                    results[key] = result
                except Exception as e:
                    print(f"Fehler bei {key} search: {str(e)}")
                    results[key] = []

            return results

    def _semantic_search(self, query: str, k: int) -> List[Dict[str, Any]]:
        """Führt semantische Suche durch"""
        try:
            # In FAISS verwenden wir direkt similarity_search statt similarity_search_with_relevance_scores
            results = self.vectorstore.similarity_search(
                query, 
                k=k
            )
            if not results:
                return []
                
            processed_results = []
            for doc in results:
                metadata = getattr(doc, "metadata", {})
                # Berechne einen Score basierend auf Kontext, da wir keine direkten Scores haben
                context_score = self._calculate_context_relevance(doc, query)
                
                processed_results.append({
                    "content": doc.page_content,
                    "score": float(context_score),  # Verwende Kontext-Score als Haupt-Score
                    "type": "semantic",
                    "metadata": metadata,
                })
            
            return processed_results
            
        except Exception as e:
            print(f"Fehler in semantic_search: {str(e)}")
            traceback.print_exc()  # Füge detaillierten Traceback hinzu
            return []
    def _keyword_search(self, query: str, k: int) -> List[Dict[str, Any]]:
        """Führt verbesserte Keyword-basierte Suche mit BM25 durch, mit Vorverarbeitung"""
        try:
            # Vorverarbeitung der Query
            preprocessed_query = self._preprocess_text(query)
            scores = self.bm25.get_scores(preprocessed_query)
            
            # Verwende self.documents anstelle von self.bm25.corpus
            scored_docs = sorted(
                zip(self.documents, scores),  # self.documents enthält die Dokumente
                key=lambda x: x[1],
                reverse=True
            )[:k]

            results = [
                {
                    "content": doc.page_content,
                    "score": float(score),
                    "type": "keyword",
                    "metadata": doc.metadata,
                    "quality_metrics": self._calculate_quality_metrics(doc.page_content),
                }
                for doc, score in scored_docs
            ]

            return results

        except Exception as e:
            print(f"Fehler in keyword search: {str(e)}")
            return []

    def _graph_search(self, query: str, k: int) -> List[Dict[str, Any]]:
        """Führt verbesserte Graph-basierte Suche durch"""
        try:
            query_embedding = self._get_or_create_embedding(query)
            similarity_scores = []

            for node, data in self.chunk_graph.nodes(data=True):
                if "embedding" in data:
                    similarity = self._calculate_cosine_similarity(
                        query_embedding, data["embedding"]
                    )
                    if similarity > self.minimum_similarity_threshold:
                      similarity_scores.append((node, similarity))

            # Sortiere nach Ähnlichkeit und wähle die Top-k Knoten aus
            similarity_scores.sort(key=lambda x: x[1], reverse=True)
            top_k_nodes = similarity_scores[:k]

            results = []
            for node, score in top_k_nodes:
                # Führe PageRank um den ausgewählten Knoten aus
                pagerank_scores = nx.pagerank(
                    self.chunk_graph, personalization={node: 1}, max_iter=200, tol=1e-06
                )
                # Wähle die Top-Dokumente basierend auf PageRank aus
                top_pagerank_nodes = sorted(
                    pagerank_scores, key=pagerank_scores.get, reverse=True
                )[:k]
                # Füge die Ergebnisse basierend auf PageRank hinzu
                results.extend(
                    [
                        {
                            "content": self.chunk_graph.nodes[n]["content"],
                            "score": float(pagerank_scores[n]),
                            "type": "graph",
                            "metadata": self.chunk_graph.nodes[n].get("metadata", {}),
                            "quality_metrics": self.chunk_graph.nodes[n].get(
                                "quality_metrics", {}
                            ),
                        }
                        for n in top_pagerank_nodes
                        if n in self.chunk_graph.nodes
                    ]
                )

            # Entferne Duplikate, falls durch PageRank und direkte Ähnlichkeitssuche verursacht
            unique_results = []
            seen_contents = set()
            for result in results:
                if result["content"] not in seen_contents:
                    unique_results.append(result)
                    seen_contents.add(result["content"])

            return unique_results[:k]  # Begrenze auf k Ergebnisse

        except Exception as e:
            print(f"Fehler in graph search: {str(e)}")
            return []

    def _statistical_search(self, query: str, k: int) -> List[Dict[str, Any]]:
        """Führt statistische Suche basierend auf TF-IDF durch, mit Vorverarbeitung"""
        try:
            # Vorverarbeitung der Query
            preprocessed_query = self._preprocess_text(query)
            
            # Erstelle einen leeren Vektor mit der richtigen Größe für die TF-IDF-Matrix
            query_vector = np.zeros(self.sparse_matrices["tf_idf"].shape[1])

            # Erstelle Query-Vektor (nur für Terme, die im Vokabular vorhanden sind)
            for term in preprocessed_query:
                if term in self.term_to_index:
                    term_idx = self.term_to_index[term]
                    query_vector[term_idx] = 1  # TF-IDF-Wert für die Query könnte hier auch berechnet werden

            # Berechne Kosinus-Ähnlichkeit mit Sparse-Matrix-Multiplikation
            similarity_scores = self.sparse_matrices["tf_idf"].dot(query_vector)

            # Finde die Top-k Dokumente
            top_k_indices = np.argpartition(similarity_scores, -k)[-k:]
            top_k_indices = top_k_indices[np.argsort(similarity_scores[top_k_indices])][::-1]

            documents = self._get_all_documents()
            results = []
            for idx in top_k_indices:
                if similarity_scores[idx] > 0:
                    doc = documents[idx]
                    results.append({
                        "content": doc.page_content,
                        "score": float(similarity_scores[idx]),
                        "type": "statistical",
                        "metadata": doc.metadata,
                    })

            return results

        except Exception as e:
            print(f"Fehler in statistical search: {str(e)}")
            return []
    def _calculate_hierarchy_distribution(self, results: List[Dict]) -> Dict[int, float]:
        try:
            level_counts = defaultdict(int)
            total_results = max(len(results), 1)  # Verhindere Division durch 0
            
            for result in results:
                metadata = result.get("metadata", {})
                level = metadata.get("hierarchy_level", 3)
                level_counts[level] += 1
            
            distribution = {
                level: count/total_results
                for level, count in level_counts.items()
            }
            
            return distribution
            
        except Exception as e:
            print(f"Fehler bei Hierarchie-Distribution: {str(e)}")
            return {}
    def _calculate_context_relevance(self, doc: Document, query: str) -> float:
        """Berechnet die Kontext-Relevanz basierend auf Hierarchie und Nachbar-Chunks"""
        try:
            metadata = doc.metadata
            
            # Prüfe Nachbar-Chunks
            neighboring_relevance = 0.0
            if "neighboring_chunks" in metadata:
                neighbors = metadata["neighboring_chunks"]
                for neighbor_id in [neighbors.get("prev"), neighbors.get("next")]:
                    if neighbor_id is not None:
                        neighbor_doc = self._get_document_by_index(neighbor_id)
                        if neighbor_doc:
                            similarity = self._calculate_cosine_similarity(
                                self._get_or_create_embedding(neighbor_doc.page_content),
                                self._get_or_create_embedding(query)
                            )
                            neighboring_relevance += similarity
                            
            neighboring_relevance = neighboring_relevance / 2 if neighboring_relevance > 0 else 0
            
            # Kombiniere mit Hierarchie-Information
            hierarchy_boost = self.hierarchy_weights.get(
                f'level_{metadata.get("hierarchy_level", 3)}', 
                0.6
            )
            
            return (neighboring_relevance * 0.3 + hierarchy_boost * 0.7)
            
        except Exception as e:
            print(f"Fehler bei Kontext-Relevanz Berechnung: {str(e)}")
            return 0.5
    
    def _evaluate_stage_quality(self, results: Dict[str, List[Dict]], stage: int) -> Dict[str, float]:
        """Evaluiert die Qualität der Ergebnisse einer Suchstufe"""
        try:
            quality_metrics = {
                "avg_score": 0.0,
                "relevance": 0.0,
                "diversity": 0.0,
                "confidence": 0.0
            }
            
            if not results:
                return quality_metrics
                
            # Berechne durchschnittliche Scores
            scores = []
            for method, method_results in results.items():
                if method_results:
                    method_scores = [r.get("score", 0) for r in method_results]
                    scores.extend(method_scores)
                    
            if scores:
                quality_metrics["avg_score"] = np.mean(scores)
                quality_metrics["confidence"] = len(scores) / (10 * len(results))  # Normalisierte Konfidenz
                
            # Berechne Diversität
            all_contents = set()
            for method_results in results.values():
                for result in method_results:
                    all_contents.add(result.get("content", ""))
            quality_metrics["diversity"] = len(all_contents) / (sum(len(r) for r in results.values()) + 1e-6)
            
            return quality_metrics
            
        except Exception as e:
            print(f"Fehler bei Qualitätsevaluation: {str(e)}")
            return quality_metrics

    def _adjust_weights_based_on_quality(
        self, 
        weights: Dict[str, float], 
        quality_metrics: Dict[str, float],
        query_type: str
    ) -> Dict[str, float]:
        """Passt die Gewichte basierend auf Qualitätsmetriken und Query-Typ an"""
        try:
            adjusted_weights = weights.copy()
            
            # Anpassung basierend auf Qualität
            if quality_metrics["avg_score"] < 0.5:
                adjusted_weights["semantic"] *= 1.2
                adjusted_weights["statistical"] *= 1.1
            
            if quality_metrics["diversity"] < 0.3:
                adjusted_weights["graph"] *= 1.2
                
            # Anpassung basierend auf Query-Typ
            if query_type == "keyword":
                adjusted_weights["keyword"] *= 1.3
                adjusted_weights["semantic"] *= 0.8
            elif query_type == "natural":
                adjusted_weights["semantic"] *= 1.3
                adjusted_weights["keyword"] *= 0.8
            
            # Normalisiere Gewichte
            total = sum(adjusted_weights.values())
            return {k: v/total for k, v in adjusted_weights.items()}
            
        except Exception as e:
            print(f"Fehler bei Gewichtsanpassung: {str(e)}")
            return weights

    def _detect_query_type(self, query: str) -> str:
        """Erkennt den Typ der Query"""
        try:
            # Frage-Erkennung
            question_words = ["was", "wie", "wo", "wann", "warum", "wer", "welche"]
            if any(query.lower().startswith(w) for w in question_words):
                return "natural"
                
            # Keyword-Erkennung
            words = query.split()
            if len(words) <= 3 and all(w[0].isupper() for w in words if w):
                return "keyword"
                
            # Komplexere natürlichsprachliche Query-Erkennung
            doc = self.nlp(query)
            if any(token.dep_ in ['ROOT', 'nsubj', 'dobj'] for token in doc):
                return "natural"
                
            return "keyword"
            
        except Exception as e:
            print(f"Fehler bei Query-Typ-Erkennung: {str(e)}")
            return "keyword"

    def _apply_quality_filter(self, results: Dict[str, List[Dict]], thresholds: Optional[Dict[str, float]] = None) -> Dict[str, List[Dict]]:
        """Wendet Qualitätsfilter auf die Suchergebnisse an"""
        if thresholds is None:
            thresholds = {
                "semantic": 0.4,
                "keyword": 0.1,
                "graph": 0.1,
                "statistical": 0.05
            }
        
        filtered_results = {}
        for method, method_results in results.items():
            threshold = thresholds.get(method, 0.1)
            filtered_results[method] = [
                r for r in method_results
                if r.get("score", 0) >= threshold
            ]
        
        return filtered_results

    # 4. Hilfsfunktionen
    def _get_all_documents(self) -> List[Document]:
        """Extrahiert alle Dokumente aus dem Vector Store"""
        try:
            documents = []
            if hasattr(self.vectorstore, "docstore") and hasattr(
                self.vectorstore, "index_to_docstore_id"
            ):
                for doc_id in self.vectorstore.index_to_docstore_id.values():
                    doc = self.vectorstore.docstore.search(doc_id)
                    if isinstance(doc, Document):
                        documents.append(doc)
                    elif isinstance(doc, (str, bytes)):
                        documents.append(Document(page_content=str(doc), metadata={}))
            return documents
        except Exception as e:
            print(f"Fehler beim Laden der Dokumente: {str(e)}")
            return []

    def _get_document_by_index(self, idx: int) -> Optional[Document]:
        """Holt ein Dokument anhand des Index"""
        try:
            if (
                hasattr(self.vectorstore, "docstore")
                and hasattr(self.vectorstore, "index_to_docstore_id")
                and idx in self.vectorstore.index_to_docstore_id
            ):
                doc_id = self.vectorstore.index_to_docstore_id[idx]
                doc = self.vectorstore.docstore.search(doc_id)
                return doc
            return None
        except Exception as e:
            print(f"Fehler beim Abrufen des Dokuments: {str(e)}")
            return None

    def _get_or_create_embedding(self, text: str) -> np.ndarray:
        """Holt oder erstellt Embedding mit Caching"""
        # Konvertiere das Embedding in einen hashbaren Typ für das Caching
        embedding_hashable = text  # Verwende den Text selbst als Schlüssel
        if embedding_hashable in self.embedding_cache:
            return self.embedding_cache[embedding_hashable]

        try:
            embedding = self.vectorstore.embedding_function.embed_query(text)
            # Stellen Sie sicher, dass das Embedding ein NumPy-Array ist
            embedding = np.array(embedding)

            self.embedding_cache[embedding_hashable] = embedding

            # Cache-Größe begrenzen
            if len(self.embedding_cache) > self.cache_size:
                self.embedding_cache.pop(next(iter(self.embedding_cache)))

            return embedding
        except Exception as e:
            print(f"Fehler bei Embedding-Erstellung: {str(e)}")
            return np.zeros(768)  # Fallback mit Null-Vektor

    def _calculate_cosine_similarity(
        self, vec1: np.ndarray, vec2: np.ndarray
    ) -> float:
        """Berechnet Kosinus-Ähnlichkeit zwischen Vektoren"""
        try:
            if vec1.ndim == 1:
                vec1 = vec1.reshape(1, -1)
            if vec2.ndim == 1:
                vec2 = vec2.reshape(1, -1)

            dot_product = np.dot(vec1, vec2.T)
            norm_vec1 = np.linalg.norm(vec1, axis=1, keepdims=True)
            norm_vec2 = np.linalg.norm(vec2, axis=1, keepdims=True)
            similarity = dot_product / (norm_vec1 * norm_vec2.T)
            return similarity[0, 0]
        except Exception as e:
            print(f"Fehler bei Ähnlichkeitsberechnung: {str(e)}")
            return 0.0

    # 5. Performance und Caching
    def _update_cache(self, key: str, results: List[Dict]):
        """Aktualisiert Result-Cache"""
        try:
            if len(self.result_cache) >= self.cache_size:
                self.result_cache.pop(next(iter(self.result_cache)))
            self.result_cache[key] = results
        except Exception as e:
            print(f"Fehler beim Cache-Update: {str(e)}")

    def _update_performance_metrics(
        self,
        query: str,
        results: List[Dict],
        start_time: datetime
    ):
        try:
            duration = (datetime.now() - start_time).total_seconds()
            
            # Erweiterte Metriken
            metrics = {
                "query_length": len(query.split()),
                "num_results": len(results),
                "processing_time": duration,
                "average_score": np.mean([r["score"] for r in results]) if results else 0,
                "cache_hit_rate": len(self.result_cache) / self.cache_size,
                "hierarchy_distribution": self._calculate_hierarchy_distribution(results),
                "source_distribution": self._calculate_source_distribution(results)
            }
            
            for key, value in metrics.items():
                self.performance_metrics[key].append(value)
                
            # Begrenze Historie
            if len(next(iter(self.performance_metrics.values()))) > 1000:
                for key in self.performance_metrics:
                    self.performance_metrics[key] = self.performance_metrics[key][-1000:]
        
        except Exception as e:
            print(f"Fehler beim Metrik-Update: {str(e)}")

    def _calculate_adaptive_weights(
        self, query: str, context: Optional[Dict], initial_weights: Optional[Dict[str, float]] = None
    ) -> Dict[str, float]:
        """Berechnet adaptive Gewichte basierend auf Query, Kontext und initialen Gewichten"""
        weights = initial_weights if initial_weights is not None else self.search_weights.copy()

        try:
            # Vorverarbeitung der Query
            preprocessed_query = self._preprocess_text(query)
            terms = preprocessed_query

            rare_terms = sum(1 for term in terms if self.term_frequency[term] < 5)

            if rare_terms > 0:
                # Erhöhe Gewicht für Keyword-Suche und statistische Suche bei seltenen Begriffen
                weights["keyword"] += 0.05
                weights["statistical"] += 0.05
                weights["semantic"] -= 0.1 # Reduziere semantische Suche

            if len(terms) > 5:
                # Erhöhe semantische Gewichtung bei längeren Queries
                weights["semantic"] += 0.1
                weights["keyword"] -= 0.05
                weights["statistical"] -= 0.05
            
            # Einfache Heuristik für Query-Typ
            if any(keyword in query.lower() for keyword in ["was", "wie", "wo", "wann", "warum"]):
                weights["semantic"] += 0.05
                weights["keyword"] -= 0.025
                weights["graph"] -= 0.025
            elif rare_terms > 0:
                weights["keyword"] += 0.05
                weights["statistical"] += 0.05
                weights["semantic"] -= 0.1

            # Kontext-basierte Anpassungen (Beispiel)
            if context and context.get("active_themes"):
                weights["context"] += 0.1
                weights["graph"] += 0.05
                weights["semantic"] -= 0.1
                weights["keyword"] -= 0.025
                weights["statistical"] -= 0.025

            # Normalisiere Gewichte
            total = sum(weights.values())
            normalized_weights = {k: v / total for k, v in weights.items()}

            return normalized_weights

        except Exception as e:
            print(f"Fehler bei Gewichtsanpassung: {str(e)}")
            return weights

    def _combine_all_results(
        self,
        search_results: Dict[str, List[Dict]],
        query: str,
        context: Optional[Dict],
        weights: Dict[str, float],
    ) -> List[Dict[str, Any]]:
        try:
            combined_scores = defaultdict(
                lambda: {
                    "score": 0.0,
                    "sources": [],
                    "metadata": {},
                    "quality_metrics": {},
                    "relevance_score": 0.0,
                    "content_quality": 0.0,
                    "hierarchy_boost": 0.0
                }
            )

            for search_type, results in search_results.items():
                if search_type not in weights:
                    continue
                    
                weight = weights[search_type]
                for result in results:
                    content = result.get("content", "")
                    if not content:
                        continue
                        
                    entry = combined_scores[content]
                    
                    # Hierarchie-Boost basierend auf Level
                    hierarchy_level = result.get("metadata", {}).get("hierarchy_level", 3)
                    hierarchy_boost = self.hierarchy_weights.get(f'level_{hierarchy_level}', 0.6)
                    
                    # Berechne normalisierte Scores
                    base_score = result.get("score", 0)
                    quality_score = self._calculate_quality_score(result, query)
                    relevance_score = self._calculate_relevance_score(result, query)
                    
                    # Kombiniere Scores mit Gewichtung und Hierarchie
                    weighted_score = (
                        base_score * 0.4 +
                        quality_score * 0.3 +
                        relevance_score * 0.3
                    ) * weight * hierarchy_boost
                    
                    entry["score"] += weighted_score
                    entry["hierarchy_boost"] = hierarchy_boost
                    entry["sources"].append(search_type)
                    entry["metadata"].update(result.get("metadata", {}))
                    entry["quality_metrics"].update(result.get("quality_metrics", {}))
                    entry["relevance_score"] = max(entry["relevance_score"], relevance_score)
                    entry["content_quality"] = max(entry["content_quality"], quality_score)

            # Erstelle sortierte Liste mit Diversity-Boost
            results_list = []
            seen_sources = set()
            for content, data in combined_scores.items():
                # Diversity Boost für Ergebnisse aus verschiedenen Quellen
                source_diversity = len(set(data["sources"])) / len(weights)
                final_score = data["score"] * (1 + 0.2 * source_diversity)
                
                results_list.append({
                    "content": content,
                    "score": final_score,
                    "sources": data["sources"],
                    "metadata": data["metadata"],
                    "quality_metrics": data["quality_metrics"],
                    "relevance_score": data["relevance_score"],
                    "content_quality": data["content_quality"]
                })
                
                # Aktualisiere gesehene Quellen
                seen_sources.update(data["sources"])

            # Sortiere nach Score und wende Diversity-Filter an
            return sorted(
                results_list,
                key=lambda x: (
                    x["score"],  # Primär nach Score
                    x["relevance_score"],  # Sekundär nach Relevanz
                    x["content_quality"]  # Tertiär nach Qualität
                ),
                reverse=True
            )

        except Exception as e:
            print(f"Fehler beim Kombinieren der Ergebnisse: {str(e)}")
            return []
    def _calculate_quality_score(self, result: Dict, query: str) -> float:
        """Berechnet einen Qualitätsscore für ein Ergebnis"""
        try:
            # Basis-Qualitätsmetriken
            metrics = result.get("quality_metrics", {})
            if not metrics:
                return 0.5  # Default Score
                
            # Kombiniere verschiedene Qualitätsaspekte
            scores = [
                metrics.get("information_density", 0.5) * 0.3,  # Informationsdichte
                metrics.get("structure_score", 0.5) * 0.2,      # Strukturqualität
                metrics.get("length_factor", 0.5) * 0.2,        # Längenoptimalität
                metrics.get("term_diversity", 0.5) * 0.3        # Begriffsdiversität
            ]
            
            return sum(scores)
            
        except Exception as e:
            print(f"Fehler bei Qualitätsberechnung: {str(e)}")
            return 0.5

    def _calculate_relevance_score(self, result: Dict, query: str) -> float:
        """Berechnet einen Relevanz-Score basierend auf Query-Matching"""
        try:
            content = result.get("content", "").lower()
            query_terms = set(self._preprocess_text(query))
            content_terms = set(self._preprocess_text(content))
            
            # Berechne Term-Overlap
            if not query_terms:
                return 0.5
                
            overlap = len(query_terms & content_terms) / len(query_terms)
            
            # Bonus für exakte Matches
            exact_match_bonus = 0.2 if query.lower() in content else 0
            
            return min(overlap + exact_match_bonus, 1.0)
            
        except Exception as e:
            print(f"Fehler bei Relevanzberechnung: {str(e)}")
            return 0.5

    def _add_graph_edges_improved(self, graph: nx.DiGraph, documents: List[Document]):
        """Verbesserte Methode zum Hinzufügen von Kanten zwischen ähnlichen Dokumenten im Graphen"""
        try:
            for i, doc1 in enumerate(documents):
                if not doc1:
                    continue
                
                for j, doc2 in enumerate(documents):
                    if i == j or not doc2:
                        continue

                    if i in graph and j in graph:
                        # Erweitere Ähnlichkeitsberechnung um Kontextinformationen
                        content_similarity = self._calculate_cosine_similarity(
                            graph.nodes[i]["embedding"], graph.nodes[j]["embedding"]
                        )

                        # Berücksichtige thematische Ähnlichkeit
                        thematic_similarity = self._calculate_thematic_similarity(doc1, doc2)

                        # Berücksichtige Qualitätsmetriken
                        quality_similarity = self._calculate_quality_similarity(
                            graph.nodes[i]["quality_metrics"], graph.nodes[j]["quality_metrics"]
                        )

                        # Gewichtete Kombination der Ähnlichkeiten
                        similarity = (
                            0.6 * content_similarity + 0.3 * thematic_similarity + 0.1 * quality_similarity
                        )

                        if similarity > self.minimum_similarity_threshold:
                            graph.add_edge(i, j, weight=float(similarity))

        except Exception as e:
            print(f"Fehler beim Hinzufügen von Kanten zum Graphen: {str(e)}")
            
    def _calculate_thematic_similarity(self, doc1: Document, doc2: Document) -> float:
        """Berechnet die thematische Ähnlichkeit zwischen zwei Dokumenten"""
        try:
            # Verwende die Themeninformationen aus dem Thematic Tracker, falls verfügbar
            doc1_themes = doc1.metadata.get("themes", [])
            doc2_themes = doc2.metadata.get("themes", [])

            if not doc1_themes or not doc2_themes:
                return 0.0

            # Jaccard-Ähnlichkeit der Themensets
            intersection = len(set(doc1_themes) & set(doc2_themes))
            union = len(set(doc1_themes) | set(doc2_themes))
            
            if union == 0:
                return 0.0
            
            return float(intersection / union)

        except Exception as e:
            print(f"Fehler bei der Berechnung der thematischen Ähnlichkeit: {str(e)}")
            return 0.0

    def _calculate_quality_similarity(self, metrics1: Dict[str, float], metrics2: Dict[str, float]) -> float:
        """Berechnet die Ähnlichkeit der Qualitätsmetriken zwischen zwei Dokumenten"""
        try:
            if not metrics1 or not metrics2:
                return 0.0

            common_metrics = set(metrics1.keys()) & set(metrics2.keys())
            if not common_metrics:
                return 0.0

            # Berechne den durchschnittlichen absoluten Unterschied für gemeinsame Metriken
            diffs = [
                abs(metrics1[metric] - metrics2[metric])
                for metric in common_metrics
            ]
            avg_diff = np.mean(diffs)

            # Ähnlichkeit ist invers zum Unterschied (1 - diff)
            return float(1 - avg_diff)

        except Exception as e:
            print(f"Fehler bei der Berechnung der Qualitätsähnlichkeit: {str(e)}")
            return 0.0

    def _calculate_quality_metrics(self, content: str) -> Dict[str, float]:
        """Berechnet Qualitätsmetriken für einen Text"""
        try:
            # Vorverarbeitung des Textes
            preprocessed_content = self._preprocess_text(content)
            words = preprocessed_content
            unique_words = set(words)
            
            # Extrahiere Sätze mit spaCy
            doc = self.nlp(content)
            sentences = [sent.text for sent in doc.sents]

            metrics = {
                "information_density": len(unique_words) / max(len(words), 1),
                "structure_score": 1.0 / (1.0 + abs(len(words) / max(len(sentences), 1) - 20)),
                "length_factor": 1 - abs(len(words) - 150) / 150,
                "term_diversity": len(unique_words) / max(len(self.term_frequency), 1),
            }

            # Gesamtqualität
            metrics["quality_score"] = np.mean(list(metrics.values()))

            return {k: float(v) for k, v in metrics.items()}

        except Exception as e:
            print(f"Fehler bei Qualitätsberechnung: {str(e)}")
            return {
                "quality_score": 0.5,
                "information_density": 0.5,
                "structure_score": 0.5,
                "length_factor": 0.5,
                "term_diversity": 0.5,
            }
    def _calculate_source_distribution(self, results: List[Dict[str, Any]]) -> Dict[str, float]:
        """Berechnet die Verteilung der Quellen in den Suchergebnissen"""
        try:
            source_counts = defaultdict(int)
            total_results = len(results)
            
            for result in results:
                source = result.get('metadata', {}).get('source', 'unknown')
                source_counts[source] += 1
            
            if total_results == 0:
                return {}
                
            return {source: count/total_results 
                    for source, count in source_counts.items()}
                    
        except Exception as e:
            print(f"Fehler bei Source-Distribution Berechnung: {str(e)}")
            return {}
    def _cleanup_cache(self):
        """Bereinigt abgelaufene Cache-Einträge"""
        try:
            current_time = datetime.now()
            if (current_time - self.last_cache_cleanup).total_seconds() > 300:  # Alle 5 Minuten
                expired_keys = [
                    key for key, (timestamp, _) in self.result_cache.items()
                    if (current_time - timestamp).total_seconds() > self.cache_ttl
                ]
                for key in expired_keys:
                    del self.result_cache[key]
                self.last_cache_cleanup = current_time
                
        except Exception as e:
            print(f"Fehler bei Cache-Bereinigung: {str(e)}")

    def _preprocess_text(self, text: str) -> List[str]:
        """Wendet verbessertes Preprocessing mit Lemmatisierung, Stopword-Entfernung und POS-Filterung an"""
        doc = self.nlp(text.lower())
        return [
            token.lemma_ for token in doc 
            if not token.is_stop and not token.is_punct and token.pos_ in ("NOUN", "VERB", "ADJ", "ADV")
        ]
from typing import Dict, Any, List, Optional
from datetime import datetime
import numpy as np
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

class ResponseGenerator:
    """Enhanced response generation with dynamic templates"""
    
    def __init__(self, llm, embeddings_model):
        self.llm = llm
        self.embeddings_model = embeddings_model
        self.response_history = []
        
        # Dynamic templates for different response types
        self.response_templates = {
            # Conversation Control Templates
            "greeting": PromptTemplate(
                input_variables=["query"],
                template="""Antworte freundlich auf die Begrüßung des Nutzers.
                
                Begrüßung: {query}
                
                Antworte natürlich und einladend. Erwähne dabei, dass du für Fragen zur Wirtschaftsprüfung zur Verfügung stehst."""
            ),
            
            "farewell": PromptTemplate(
                input_variables=["query"],
                template="""Antworte freundlich auf die Verabschiedung des Nutzers.
                
                Verabschiedung: {query}
                
                Verabschiede dich höflich und biete an, dass der Nutzer jederzeit mit weiteren Fragen zurückkommen kann."""
            ),
            
            "gratitude": PromptTemplate(
                input_variables=["query"],
                template="""Reagiere angemessen auf den Dank des Nutzers.
                
                Äußerung: {query}
                
                Antworte bescheiden und biete weitere Hilfe an."""
            ),
            
            "acknowledgment": PromptTemplate(
                input_variables=["query"],
                template="""Reagiere auf die Bestätigung des Nutzers.
                
                Äußerung: {query}
                
                Gib eine kurze, bestätigende Antwort und frage optional, ob weitere Fragen bestehen."""
            ),
            
            "positive_feedback": PromptTemplate(
                input_variables=["query"],
                template="""Reagiere auf das positive Feedback des Nutzers.
                
                Feedback: {query}
                
                Bedanke dich für das Feedback und biete weitere Unterstützung an."""
            ),
            
            "negative_feedback": PromptTemplate(
                input_variables=["query"],
                template="""Reagiere konstruktiv auf das negative Feedback des Nutzers.
                
                Feedback: {query}
                
                Entschuldige dich und biete an, die Information anders zu erklären oder weitere Klärung zu geben."""
            ),
            
            # Follow-up und Multi-Intent Templates
            "follow_up": PromptTemplate(
                input_variables=["query", "previous_response", "context"],
                template="""Beantworte die Nachfrage des Nutzers unter Berücksichtigung der vorherigen Antwort.
                
                Vorherige Antwort: {previous_response}
                Nachfrage: {query}
                Zusätzlicher Kontext: {context}
                
                Kläre die offenen Punkte und stelle sicher, dass die Erklärung verständlich ist."""
            ),
            
            "multi_intent": PromptTemplate(
                input_variables=["query", "intents", "context"],
                template="""Beantworte alle Aspekte der Nutzeranfrage.
                
                Anfrage: {query}
                Erkannte Intents: {intents}
                Kontext: {context}
                
                Gehe auf jeden Aspekt einzeln ein und stelle eine kohärente Antwort zusammen."""
            ),
            
            # Information Templates
            "information": PromptTemplate(
                input_variables=["context", "query", "key_topics"],
                template="""Basierend auf dem gegebenen Kontext, beantworte die Frage klar und präzise.
                Fokussiere auf die wichtigsten Fakten und Konzepte.

                Kontext: {context}
                Frage: {query}
                Wichtige Themen: {key_topics}

                Konzentriere dich auf diese Aspekte:
                - Klare Definition und Erklärung
                - Wichtigste Fakten
                - Konkrete Beispiele wo sinnvoll
                
                Antworte in einem professionellen, aber zugänglichen Stil."""
            ),
            
            "process": PromptTemplate(
                input_variables=["context", "query", "key_topics"],
                template="""Erkläre den angefragten Prozess oder Ablauf klar und strukturiert.

                Kontext: {context}
                Frage: {query}
                Wichtige Aspekte: {key_topics}

                Strukturiere die Antwort wie folgt:
                - Kurze Prozessübersicht
                - Wichtigste Schritte
                - Kritische Aspekte oder Hinweise
                
                Verwende eine klare, schrittweise Struktur."""
            ),
            
            "comparison": PromptTemplate(
                input_variables=["context", "query", "key_topics"],
                template="""Erstelle einen strukturierten Vergleich der angefragten Konzepte.

                Kontext: {context}
                Frage: {query}
                Zu vergleichende Aspekte: {key_topics}

                Strukturiere den Vergleich wie folgt:
                - Wichtigste Gemeinsamkeiten
                - Zentrale Unterschiede
                - Praktische Bedeutung
                
                Stelle die Informationen klar und ausgewogen dar."""
            ),
            
            "clarification": PromptTemplate(
                input_variables=["context", "query", "key_topics"],
                template="""Stelle die angefragten Inhalte verständlich dar und kläre mögliche Unklarheiten.

                Kontext: {context}
                Frage: {query}
                Zu klärende Aspekte: {key_topics}

                Fokussiere auf:
                - Präzise Erklärungen
                - Klärung von Missverständnissen
                - Konkrete Beispiele
                
                Stelle sicher, dass die Antwort leicht verständlich ist."""
            ),
            
            "application": PromptTemplate(
                input_variables=["context", "query", "key_topics"],
                template="""Erkläre die praktische Anwendung oder Umsetzung des angefragten Themas.

                Kontext: {context}
                Frage: {query}
                Wichtige Aspekte: {key_topics}

                Strukturiere die Antwort wie folgt:
                - Praktische Umsetzung
                - Konkrete Beispiele
                - Wichtige Hinweise
                
                Fokussiere auf die praktische Anwendbarkeit."""
            ),
            
            "meta": PromptTemplate(
                input_variables=["context", "query", "key_topics"],
                template="""Beantworte die Frage über unsere Konversation oder meine Fähigkeiten.

                Kontext: {context}
                Frage: {query}
                Relevante Aspekte: {key_topics}

                Antworte:
                - Direkt und ehrlich
                - Mit klaren Erläuterungen
                - Mit Beispielen wo hilfreich
                
                Bleibe sachlich und informativ."""
            )
        }

    def _update_response_history(self, response_text: str):
        """Aktualisiert den Antwortverlauf"""
        self.response_history.append({
            'response': response_text,
            'timestamp': datetime.now()
        })
        
        # Begrenze Historie
        if len(self.response_history) > 10:
            self.response_history.pop(0)

    def generate_response(self, query: str, context: List[Dict], intent_info: Dict[str, Any]) -> Dict[str, Any]:
        """Generates a response based on intent and context"""
        try:
            # Validiere intent_info
            if not isinstance(intent_info, dict):
                intent_info = {'intent': 'information', 'confidence': 0.5, 'response_type': 'information'}

            # Handle conversation control intents
            if intent_info.get('intent') in {'greeting', 'farewell', 'gratitude', 'acknowledgment'}:
                return self._handle_conversation_response(query, intent_info)

            # Handle multi-intent responses
            if intent_info.get('multi_intent', False):
                return self._handle_multi_intent_response(query, context, intent_info)

            # Handle follow-up responses
            if intent_info.get('is_follow_up', False):
                return self._handle_follow_up_response(query, context, intent_info)

            # Standard response generation
            return self._handle_standard_response(query, context, intent_info)

        except Exception as e:
            logger.error(f"Error in response generation: {str(e)}")
            return self._create_error_response(str(e))

    def _handle_conversation_response(self, query: str, intent_info: Dict[str, Any]) -> Dict[str, Any]:
        """Handles conversation control intents like greetings"""
        try:
            response_text = self._generate_conversation_response(query, intent_info['intent'])
            response = {
                'response': response_text,
                'metadata': {
                    'intent': intent_info,
                    'response_type': 'conversation',
                    'confidence': intent_info.get('confidence', 1.0)
                }
            }
            self._update_response_history(response)
            return response
        except Exception as e:
            logger.error(f"Error in conversation response: {str(e)}")
            return self._create_error_response(str(e))

    def _handle_multi_intent_response(self, query: str, context: List[Dict], intent_info: Dict[str, Any]) -> Dict[str, Any]:
        """Handles queries with multiple intents"""
        try:
            # Generate response for each subintent
            responses = []
            for subintent in intent_info.get('intent_sequence', []):
                sub_intent_info = {
                    'intent': subintent,
                    'key_topics': intent_info.get('key_topics', []),
                    'confidence': intent_info.get('confidence', 0.0),
                    'response_type': self.intent_categories.get(subintent, {}).get('response_type', 'information')
                }
                responses.append(self._handle_standard_response(query, context, sub_intent_info))

            # Combine responses
            combined_response = self._combine_responses(responses)
            self._update_response_history(combined_response)
            return combined_response

        except Exception as e:
            logger.error(f"Error in multi-intent response: {str(e)}")
            return self._create_error_response(str(e))

    def _handle_follow_up_response(self, query: str, context: List[Dict], intent_info: Dict[str, Any]) -> Dict[str, Any]:
        """Handles follow-up questions"""
        try:
            template = self.response_templates['follow_up']
            chain = LLMChain(llm=self.llm, prompt=template)
            
            # Get previous response
            previous_response = intent_info.get('follow_up_reference', '')
            context_str = self._prepare_context(context)
            
            response = chain.invoke({
                "query": query,
                "previous_response": previous_response,
                "context": context_str
            })
            
            response_text = response.get('text', '') if isinstance(response, dict) else str(response)
            response_text = self._clean_response(response_text)
            
            result = {
                'response': response_text,
                'metadata': {
                    'intent': intent_info,
                    'is_follow_up': True,
                    'confidence': intent_info.get('confidence', 0.0),
                    'response_type': 'follow_up'
                }
            }
            
            self._update_response_history(result)
            return result

        except Exception as e:
            logger.error(f"Error in follow-up response: {str(e)}")
            return self._create_error_response(str(e))

    def _handle_standard_response(self, query: str, context: List[Dict], intent_info: Dict[str, Any]) -> Dict[str, Any]:
        """Handles standard information-seeking queries"""
        try:
            # Select template and prepare chain
            template = self._select_template(intent_info)
            chain = LLMChain(llm=self.llm, prompt=template)
            
            # Prepare inputs
            context_str = self._prepare_context(context)
            key_topics = ", ".join(intent_info.get('key_topics', []))
            
            # Generate response
            response = chain.invoke({
                "context": context_str,
                "query": query,
                "key_topics": key_topics
            })
            
            response_text = response.get('text', '') if isinstance(response, dict) else str(response)
            response_text = self._clean_response(response_text)
            
            result = {
                'response': response_text,
                'metadata': {
                    'intent': intent_info,
                    'key_topics': intent_info.get('key_topics', []),
                    'confidence': intent_info.get('confidence', 0.0),
                    'response_type': intent_info.get('response_type', 'information')
                }
            }
            
            self._update_response_history(result)
            return result

        except Exception as e:
            logger.error(f"Error in standard response: {str(e)}")
            return self._create_error_response(str(e))
    def _select_template(self, intent_info: Dict[str, Any]) -> PromptTemplate:
        """
        Wählt das passende Template basierend auf Intent-Informationen
        
        Args:
            intent_info: Dictionary mit Intent-Informationen
            
        Returns:
            PromptTemplate: Das ausgewählte Template
        """
        try:
            # Hole den Intent-Typ
            intent_type = intent_info.get('intent', 'information')
            response_type = intent_info.get('response_type', 'information')
            
            # Versuche zuerst das spezifische Template zu bekommen
            template = self.response_templates.get(intent_type)
            
            if not template:
                # Fallback auf response_type Template
                template = self.response_templates.get(response_type)
                
            if not template:
                # Fallback auf Standard-Informationstemplate
                template = self.response_templates['information']
                logger.warning(f"Kein spezifisches Template für Intent {intent_type} gefunden, verwende Standard-Template")
            
            return template
            
        except Exception as e:
            logger.error(f"Fehler bei Template-Auswahl: {str(e)}")
            # Fallback auf Basis-Template
            return self.response_templates['information']
    def _prepare_context(self, context: List[Dict]) -> str:
        """
        Bereitet den Kontext für die Template-Verarbeitung vor
        
        Args:
            context: Liste von Kontext-Dictionaries
            
        Returns:
            str: Formatierter Kontext-String
        """
        try:
            if not context:
                return "Kein relevanter Kontext verfügbar."
                
            # Extrahiere und formatiere relevante Informationen
            context_parts = []
            for ctx in context:
                if isinstance(ctx, dict):
                    content = ctx.get('content', '')
                    if not content:
                        continue
                        
                    # Füge Metadaten hinzu, falls verfügbar
                    metadata = ctx.get('metadata', {})
                    source = metadata.get('source', '')
                    if source:
                        context_parts.append(f"[{source}] {content}")
                    else:
                        context_parts.append(content)
                elif isinstance(ctx, str):
                    context_parts.append(ctx)
                    
            # Kombiniere zu einem String
            if context_parts:
                return "\n\n".join(context_parts)
            else:
                return "Kein relevanter Kontext verfügbar."
                
        except Exception as e:
            logger.error(f"Fehler bei der Kontextvorbereitung: {str(e)}")
            return "Fehler bei der Kontextvorbereitung."
    def _clean_response(self, response: str) -> str:
        """
        Bereinigt und formatiert die Antwort
        
        Args:
            response: Rohe Antwort vom LLM
            
        Returns:
            str: Bereinigte und formatierte Antwort
        """
        try:
            # Entferne unnötige Whitespaces
            response = response.strip()
            
            # Entferne mehrfache Zeilenumbrüche
            response = "\n".join(line for line in response.splitlines() if line.strip())
            
            # Entferne "Assistant:" oder ähnliche Präfixe
            prefixes_to_remove = ["Assistant:", "AI:", "Antwort:"]
            for prefix in prefixes_to_remove:
                if response.startswith(prefix):
                    response = response[len(prefix):].strip()
            
            # Stelle sicher, dass die Antwort nicht leer ist
            if not response:
                return "Entschuldigung, ich konnte keine passende Antwort generieren."
                
            return response
            
        except Exception as e:
            logger.error(f"Fehler bei der Antwortbereinigung: {str(e)}")
            return "Fehler bei der Antwortverarbeitung."
    def _generate_conversation_response(self, query: str, intent_type: str) -> str:
        """Generates response for conversation intents using templates"""
        template = self.response_templates.get(intent_type)
        if not template:
            return "Wie kann ich Ihnen helfen?"
            
        chain = LLMChain(llm=self.llm, prompt=template)
        response = chain.invoke({"query": query})
        
        response_text = response.get('text', '') if isinstance(response, dict) else str(response)
        return self._clean_response(response_text)

    def _update_response_history(self, response: Dict[str, Any]):
        """Updates the response history"""
        if not isinstance(response, dict):
            logger.warning("Invalid response format for history update")
            return

        self.response_history.append({
            'response': response.get('response', ''),
            'metadata': response.get('metadata', {}),
            'timestamp': datetime.now()
        })
        
        # Keep history manageable
        if len(self.response_history) > 10:
            self.response_history.pop(0)

    def _combine_responses(self, responses: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Kombiniert mehrere Antworten zu einer kohärenten Gesamtantwort"""
        combined_text = []
        combined_metadata = {
            'intents': [],
            'key_topics': set(),
            'confidence': 0.0,
            'response_types': []
        }
        
        for resp in responses:
            combined_text.append(resp['response'])
            meta = resp['metadata']
            combined_metadata['intents'].append(meta.get('intent'))
            combined_metadata['key_topics'].update(meta.get('key_topics', []))
            combined_metadata['confidence'] += meta.get('confidence', 0.0)
            combined_metadata['response_types'].append(meta.get('response_type'))
        
        # Normalisiere Confidence
        if responses:
            combined_metadata['confidence'] /= len(responses)
        
        # Konvertiere Set zu List für JSON-Serialisierung
        combined_metadata['key_topics'] = list(combined_metadata['key_topics'])
        
        return {
            'response': "\n\n".join(combined_text),
            'metadata': combined_metadata
        }

    def _create_error_response(self, error_message: str) -> Dict[str, Any]:
        """Erstellt eine formatierte Fehlerantwort"""
        return {
            'response': "Entschuldigung, ich konnte keine passende Antwort generieren.",
            'metadata': {
                'error': error_message,
                'intent': 'error',
                'confidence': 0.0,
                'response_type': 'error'
            }
        }

class ResponseQualityChecker:
    def __init__(self, embeddings_model):
        self.shared_models = SharedModels()  # Verwende geteilte Modelle
        self.embeddings_model = self.shared_models.embeddings_model
        self.quality_thresholds = {
            'relevance': 0.3,
            'completeness': 0.3,
            'coherence': 0.3,
            'accuracy': 0.3
        }
        self._similarity_cache = {}

    def check_response(self, response: str, query: str, context: List[Dict], intent_info: Dict, quality_thresholds: Dict[str, float]) -> Dict[str, Any]:
        # Erweiterte Qualitätsprüfung
        if intent_info.get('is_followup'):
            relevance_threshold = 0.5
            completeness_threshold = 0.5
        else:
            relevance_threshold = 0.5
            completeness_threshold = 0.5
            
        metrics = {
            'relevance_score': self._check_relevance(response, query),
            'completeness_score': self._check_completeness(response, context),
            'coherence_score': self._check_coherence(response),
            'accuracy_score': self._check_accuracy(response, context),
            'intent_alignment': self._check_intent_alignment(response, intent_info)
        }
        
        overall_score = np.mean([
            metrics[f'{key}_score'] 
            for key in ['relevance', 'completeness', 'coherence', 'accuracy']
        ])
        
        passes_threshold = all(
            metrics[f'{key}_score'] >= threshold
            for key, threshold in self.quality_thresholds.items()
        )
        
        return {
            'passes_threshold': passes_threshold,
            'overall_score': float(overall_score),
            'metrics': metrics,
            'improvement_suggestions': self._generate_suggestions(metrics)
        }
    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """Berechnet die Ähnlichkeit zwischen zwei Texten"""
        try:
            # Nutze die Embeddings statt Spacy für die Ähnlichkeitsberechnung
            emb1 = self.embeddings_model.embed_query(text1)
            emb2 = self.embeddings_model.embed_query(text2)
            
            similarity = np.dot(emb1, emb2) / (
                np.linalg.norm(emb1) * np.linalg.norm(emb2)
            )
            return float(similarity)
        except Exception as e:
            print(f"Fehler bei Ähnlichkeitsberechnung: {str(e)}")
            return 0.5  # Fallback-Wert

    def _check_relevance(self, response: str, query: str) -> float:
        """Optimierte Relevansprüfung mit Caching"""
        cache_key = f"{query[:50]}_{response[:50]}"
        if cache_key in self._similarity_cache:
            return self._similarity_cache[cache_key]
            
        query_embedding = self.embeddings_model.embed_query(query)
        response_embedding = self.embeddings_model.embed_query(response)
        
        similarity = float(np.dot(query_embedding, response_embedding) / (
            max(np.linalg.norm(query_embedding) * np.linalg.norm(response_embedding), 1e-8)
        ))
        
        self._similarity_cache[cache_key] = similarity
        if len(self._similarity_cache) > 1000:  # Cache-Größe begrenzen
            self._similarity_cache.pop(next(iter(self._similarity_cache)))
            
        return similarity

    def _check_completeness(self, response: str, context: List[Dict]) -> float:
        """Verbesserte Vollständigkeitsprüfung"""
        try:
            nlp = spacy.load("de_core_news_sm")
            
            # Extrahiere wichtige Begriffe aus dem Kontext
            context_terms = set()
            context_entities = set()
            
            for ctx in context[:3]:
                doc = nlp(ctx['content'])
                
                # Sammle benannte Entitäten
                for ent in doc.ents:
                    context_entities.add(ent.text.lower())
                
                # Sammle wichtige Begriffe
                for token in doc:
                    if (token.pos_ in ['NOUN', 'PROPN', 'VERB'] and 
                        not token.is_stop and 
                        len(token.text) > 2):  # Filtere sehr kurze Wörter
                        context_terms.add(token.lemma_.lower())
            
            # Analysiere Antwort
            response_doc = nlp(response)
            response_terms = set()
            response_entities = set()
            
            # Sammle Entitäten aus der Antwort
            for ent in response_doc.ents:
                response_entities.add(ent.text.lower())
            
            # Sammle Begriffe aus der Antwort
            for token in response_doc:
                if (token.pos_ in ['NOUN', 'PROPN', 'VERB'] and 
                    not token.is_stop and 
                    len(token.text) > 2):
                    response_terms.add(token.lemma_.lower())
            
            # Berechne verschiedene Abdeckungsmetriken
            if not context_terms and not context_entities:
                return 0.7
            
            term_coverage = (
                len(response_terms.intersection(context_terms)) / 
                max(len(context_terms), 1)
            )
            
            entity_coverage = (
                len(response_entities.intersection(context_entities)) / 
                max(len(context_entities), 1)
            ) if context_entities else 1.0
            
            # Längenbasierte Faktoren
            min_expected_length = 30
            max_expected_length = 200
            response_length = len(response.split())
            
            length_factor = min(
                response_length / min_expected_length if response_length < min_expected_length
                else max_expected_length / response_length if response_length > max_expected_length
                else 1.0,
                1.0
            )
            
            # Kombiniere Metriken
            weights = {
                'terms': 0.4,
                'entities': 0.4,
                'length': 0.2
            }
            
            final_score = (
                term_coverage * weights['terms'] +
                entity_coverage * weights['entities'] +
                length_factor * weights['length']
            )
            
            return float(min(max(final_score, 0.3), 1.0))
            
        except Exception as e:
            print(f"Error in completeness check: {str(e)}")
            return 0.5

    def _check_coherence(self, response: str) -> float:
        """Verbesserte Kohärenzprüfung"""
        try:
            nlp = spacy.load("de_core_news_md")
            doc = nlp(response)
            
            # Sammle Sätze
            sentences = list(doc.sents)
            if len(sentences) <= 1:
                return 1.0
            
            coherence_scores = []
            
            # Berechne Satzübergänge
            for i in range(len(sentences) - 1):
                current_sent = sentences[i]
                next_sent = sentences[i + 1]
                
                # Berechne Ähnlichkeit
                similarity = current_sent.similarity(next_sent)
                
                # Prüfe auf Konnektoren
                has_connector = any(
                    token.dep_ in ['mark', 'cc'] or
                    token.text.lower() in ['dann', 'deshalb', 'daher', 'also', 'jedoch', 'aber']
                    for token in next_sent
                )
                
                # Bonus für gute Übergänge
                if has_connector:
                    similarity = min(similarity + 0.2, 1.0)
                
                coherence_scores.append(similarity)
            
            # Berechne Gesamtkohärenz
            avg_coherence = np.mean(coherence_scores)
            
            # Berücksichtige Textlänge
            length_bonus = min(len(sentences) / 10, 0.2)  # Bonus für längere Texte
            
            final_score = min(avg_coherence + length_bonus, 1.0)
            
            return float(final_score)
            
        except Exception as e:
            print(f"Error in coherence check: {str(e)}")
            return 0.7

    def _check_accuracy(self, response: str, context: List[Dict]) -> float:
        """Verbesserte Genauigkeitsprüfung"""
        try:
            if not context:
                return 0.7
            
            response_embedding = self.embeddings_model.embed_query(response)
            
            # Berechne verschiedene Ähnlichkeitsmetriken
            similarities = []
            fact_consistency = []
            
            nlp = spacy.load("de_core_news_sm")
            response_doc = nlp(response)
            response_facts = self._extract_facts(response_doc)
            
            for ctx in context[:3]:
                # Embedding-basierte Ähnlichkeit
                ctx_embedding = self.embeddings_model.embed_query(ctx['content'])
                emb_similarity = np.dot(response_embedding, ctx_embedding) / (
                    np.linalg.norm(response_embedding) * np.linalg.norm(ctx_embedding)
                )
                similarities.append(emb_similarity)
                
                # Faktenbasierte Konsistenz
                ctx_doc = nlp(ctx['content'])
                ctx_facts = self._extract_facts(ctx_doc)
                
                if ctx_facts and response_facts:
                    fact_overlap = len(response_facts.intersection(ctx_facts)) / len(ctx_facts)
                    fact_consistency.append(fact_overlap)
            
            # Kombiniere Metriken
            embedding_score = max(similarities) if similarities else 0.0
            fact_score = max(fact_consistency) if fact_consistency else 1.0
            
            # Gewichtete Kombination
            weights = {'embedding': 0.7, 'facts': 0.3}
            combined_score = (
                embedding_score * weights['embedding'] +
                fact_score * weights['facts']
            )
            
            # Skalierung und Mindestgarantie
            if len(response.split()) > 20:
                combined_score = max(combined_score, 0.4)
            
            return float(min(combined_score * 1.2, 1.0))
            
        except Exception as e:
            print(f"Error in accuracy check: {str(e)}")
            return 0.6

    def _extract_facts(self, doc) -> Set[str]:
        """Extrahiert Fakten aus einem Spacy Doc"""
        facts = set()
        for sent in doc.sents:
            for ent in sent.ents:
                facts.add(f"{ent.label_}:{ent.text.lower()}")
            
            for token in sent:
                if token.dep_ in ['nsubj', 'dobj'] and token.head.pos_ == 'VERB':
                    facts.add(f"{token.dep_}:{token.text.lower()}:{token.head.text.lower()}")
        
        return facts

    def _check_intent_alignment(self, response: str, intent_info: Dict) -> float:
        intent_type = intent_info['intent']
        
        intent_markers = {
            'definition': ['ist', 'bezeichnet', 'bedeutet', 'definiert'],
            'process': ['zunächst', 'dann', 'schließlich', 'folgt'],
            'comparison': ['während', 'hingegen', 'im Gegensatz', 'ähnlich'],
            'analysis': ['daher', 'folglich', 'zeigt sich', 'resultiert']
        }
        
        markers = intent_markers.get(intent_type, [])
        if not markers:
            return 1.0
            
        response_lower = response.lower()
        marker_count = sum(1 for marker in markers if marker in response_lower)
        
        return float(marker_count / len(markers))

    def _generate_suggestions(self, metrics: Dict[str, float]) -> List[str]:
        suggestions = []
        
        if metrics['relevance_score'] < self.quality_thresholds['relevance']:
            suggestions.append(
                "Die Antwort könnte besser auf die spezifische Frage eingehen."
            )
        
        if metrics['completeness_score'] < self.quality_thresholds['completeness']:
            suggestions.append(
                "Wichtige Aspekte aus dem Kontext fehlen in der Antwort."
            )
        
        if metrics['coherence_score'] < self.quality_thresholds['coherence']:
            suggestions.append(
                "Die logische Struktur der Antwort könnte verbessert werden."
            )
        
        if metrics['accuracy_score'] < self.quality_thresholds['accuracy']:
            suggestions.append(
                "Die Antwort weicht zu stark vom gegebenen Kontext ab."
            )
        
        return suggestions

@dataclass
class EnhancedThemeMetadata:
    """Erweiterte Metadaten für Themen"""
    name: str
    last_mentioned: datetime
    mention_count: int = 1
    related_concepts: Set[str] = field(default_factory=set)
    referenced_laws: Set[str] = field(default_factory=set)
    answered_aspects: List[str] = field(default_factory=list)
    confidence_score: float = 1.0
    context_history: List[Dict] = field(default_factory=list)
    related_queries: List[str] = field(default_factory=list)
    concept_weights: Dict[str, float] = field(default_factory=dict)
    
class EnhancedThematicTracker:
    """Erweitertes thematisches Tracking mit Kontextbewusstsein"""
    
    def __init__(self, embeddings_model, similarity_threshold: float = 0.7):
        self.shared_models = SharedModels()
        self.embeddings_model = self.shared_models.embeddings_model
        self.current_themes: Dict[str, EnhancedThemeMetadata] = {}
        self.theme_history: List[Dict[str, EnhancedThemeMetadata]] = []
        self.decay_factor = 0.9
        self.max_history_length = 10
        self.context_window = 5
        self.theme_relations = defaultdict(list)
        self.similarity_threshold = similarity_threshold
        self._theme_embedding_cache = {}

    @lru_cache(maxsize=100)
    def _get_theme_embedding(self, theme: str) -> np.ndarray:
        """Cached Berechnung des Embeddings eines Themas"""
        if theme in self._theme_embedding_cache:
            return self._theme_embedding_cache[theme]

        # Berechnung des Embeddings basierend auf den Konzepten des Themas
        if theme in self.current_themes:
            concept_weights = self.current_themes[theme].concept_weights
            if concept_weights:
                # Gewichtete Durchschnittsberechnung der Konzept-Embeddings
                embeddings = [
                    self.embeddings_model.embed_query(concept)
                    for concept in concept_weights.keys()
                ]
                weights = np.array(list(concept_weights.values()))
                # Normalisiere die Gewichte
                weights /= weights.sum()
                weighted_avg_embedding = np.average(embeddings, axis=0, weights=weights)
                self._theme_embedding_cache[theme] = weighted_avg_embedding
                return weighted_avg_embedding

        # Fallback: Embedding basierend auf dem Themennamen
        embedding = self.embeddings_model.embed_query(theme)
        self._theme_embedding_cache[theme] = embedding
        return embedding

    def _calculate_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """Berechnet die Kosinus-Ähnlichkeit zwischen zwei Embeddings"""
        similarity = np.dot(embedding1, embedding2) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))
        return similarity
    
    def update_themes(self, message: str, detected_themes: List[str], 
                 concepts: Dict[str, Any], intent_info: Dict[str, Any],
                 context: Optional[Dict] = None) -> Dict[str, Any]:
        """Aktualisiert Themen mit erweiterter Kontextanalyse und effizienterer Ähnlichkeitsberechnung"""
        current_time = datetime.now()
        self._apply_time_decay(current_time)

        updates = {
            'new_themes': [],
            'updated_themes': [],
            'theme_shifts': [],
            'context_updates': []
        }

        # Verarbeite erkannte Themen
        for theme in detected_themes:
            is_new_theme = theme not in self.current_themes
            
            if is_new_theme:
                # Erstelle neues Thema
                self.current_themes[theme] = EnhancedThemeMetadata(
                    name=theme,
                    last_mentioned=current_time
                )
                updates['new_themes'].append(theme)
            else:
                updates['updated_themes'].append(theme)
            
            # Aktualisiere existierendes Thema
            theme_data = self.current_themes[theme]
            theme_data.mention_count += 1
            theme_data.last_mentioned = current_time
            theme_data.confidence_score = min(1.0, theme_data.confidence_score + 0.1)
            
            # Aktualisiere Konzepte und deren Gewichte
            for concept, score in concepts.get('confidence_scores', {}).items():
                theme_data.concept_weights[concept] = max(
                    score,
                    theme_data.concept_weights.get(concept, 0)
                )

            # Füge Kontextinformationen hinzu
            if context:
                theme_data.context_history.append({
                    'timestamp': current_time,
                    'context': context,
                    'intent': intent_info.get('intent') if intent_info else None
                })
                updates['context_updates'].append({
                    'theme': theme,
                    'context': context
                })

            # Optimiere Themenverschiebungs-Erkennung
            if not is_new_theme:
                if self._detect_theme_shift_optimized(theme, message):
                    updates['theme_shifts'].append(theme)

        self._archive_old_themes(current_time)
        self._update_theme_relations_optimized() # Optimierte Version

        return updates

    def _detect_theme_shift_optimized(self, theme: str, message: str) -> bool:
        """Optimierte Erkennung von Themenwechseln mit Embedding-Caching"""
        if theme not in self.current_themes:
            return True

        theme_data = self.current_themes[theme]

        # Prüfe Ähnlichkeit mit aktuellem Kontext
        if theme_data.context_history:
            message_embedding = self.embeddings_model.embed_query(message)

            # Hole das Themen-Embedding aus dem Cache oder berechne es
            theme_embedding = self._get_theme_embedding(theme)

            similarity = self._calculate_similarity(message_embedding, theme_embedding)
            return similarity < self.similarity_threshold

        return True
    
    def _update_theme_relations_optimized(self):
        """Effizientere Aktualisierung der Themenbeziehungen"""
        
        new_or_updated_themes = [
            theme for theme, data in self.current_themes.items()
            if (datetime.now() - data.last_mentioned).total_seconds() < 3600  # Nur kürzlich aktualisierte Themen
        ]

        for theme1 in new_or_updated_themes:
            theme1_data = self.current_themes[theme1]
            theme1_embedding = self._get_theme_embedding(theme1)

            for theme2 in self.current_themes:
                if theme1 != theme2:
                    theme2_data = self.current_themes[theme2]
                    theme2_embedding = self._get_theme_embedding(theme2)

                    # Berechne Ähnlichkeit der Embeddings
                    similarity = self._calculate_similarity(theme1_embedding, theme2_embedding)

                    if similarity > self.similarity_threshold:
                        # Füge Beziehung hinzu oder aktualisiere sie
                        existing_relation = next((r for r in self.theme_relations[theme1] if r['target'] == theme2), None)
                        if existing_relation:
                            existing_relation['similarity'] = similarity
                            existing_relation['timestamp'] = datetime.now()
                        else:
                            self.theme_relations[theme1].append({
                                'target': theme2,
                                'similarity': similarity,
                                'timestamp': datetime.now()
                            })

    def get_current_context(self) -> Dict[str, Any]:
        """Liefert erweiterten aktuellen Kontext"""
        context = {
            'active_themes': [],
            'theme_weights': {},
            'concept_network': {},
            'temporal_context': [],
            'theme_relations': dict(self.theme_relations)
        }
        
        total_weight = 0
        for theme, metadata in self.current_themes.items():
            # Berechne zeitbasiertes Gewicht
            time_since_mention = (
                datetime.now() - metadata.last_mentioned
            ).total_seconds()
            recency_weight = np.exp(-time_since_mention / 3600)
            frequency_weight = np.log1p(metadata.mention_count)
            
            # Kombiniere Gewichte
            weight = recency_weight * frequency_weight * metadata.confidence_score
            context['theme_weights'][theme] = weight
            total_weight += weight
            
            # Füge Theme-spezifische Informationen hinzu
            context['active_themes'].append({
                'name': theme,
                'concepts': dict(metadata.concept_weights),
                'context_history': metadata.context_history[-self.context_window:],
                'related_queries': metadata.related_queries
            })
        
        # Normalisiere Gewichte
        if total_weight > 0:
            context['theme_weights'] = {
                k: v/total_weight 
                for k, v in context['theme_weights'].items()
            }
        
        # Füge temporalen Kontext hinzu
        context['temporal_context'] = self._build_temporal_context()
        
        return context

    def _apply_time_decay(self, current_time: datetime):
        """Wendet zeitbasierte Gewichtsabnahme an"""
        for theme_data in self.current_themes.values():
            time_diff = (
                current_time - theme_data.last_mentioned
            ).total_seconds()
            
            # Berechne Decay-Faktor
            decay = self.decay_factor ** (time_diff / 3600)
            
            # Aktualisiere Scores
            theme_data.confidence_score *= decay
            theme_data.concept_weights = {
                k: v * decay 
                for k, v in theme_data.concept_weights.items()
            }
    
    def _archive_old_themes(self, current_time: datetime):
        """Archiviert alte Themen mit verbesserter Logik"""
        cutoff_time = current_time - timedelta(hours=2)
        themes_to_archive = {
            theme: data for theme, data in self.current_themes.items()
            if (
                data.last_mentioned < cutoff_time or 
                data.confidence_score < 0.1
            ) and not self._has_active_relations(theme)
        }
        
        if themes_to_archive:
            # Archiviere Themen
            self.theme_history.append(themes_to_archive)
            
            # Entferne aus aktuellen Themen
            for theme in themes_to_archive:
                del self.current_themes[theme]
            
            # Halte Historie in Grenzen
            if len(self.theme_history) > self.max_history_length:
                self.theme_history.pop(0)
    
    def _has_active_relations(self, theme: str) -> bool:
        """Prüft ob ein Thema noch aktive Beziehungen hat"""
        return any(
            theme in relation 
            for relations in self.theme_relations.values()
            for relation in relations
        )
    
    def _calculate_theme_similarity(self, 
                                  theme1_data: EnhancedThemeMetadata,
                                  theme2_data: EnhancedThemeMetadata) -> float:
        """Berechnet Ähnlichkeit zwischen Themen"""
        # Konzeptbasierte Ähnlichkeit
        common_concepts = set(theme1_data.concept_weights.keys()) & set(
            theme2_data.concept_weights.keys()
        )
        
        if not common_concepts:
            return 0.0
        
        similarity = sum(
            min(
                theme1_data.concept_weights[concept],
                theme2_data.concept_weights[concept]
            )
            for concept in common_concepts
        ) / len(common_concepts)
        
        return similarity
    
    def _build_temporal_context(self) -> List[Dict[str, Any]]:
        """Erstellt temporalen Kontext aus der Theme-Historie"""
        temporal_context = []
        
        for historical_themes in self.theme_history[-self.context_window:]:
            context_entry = {
                'timestamp': max(
                    data.last_mentioned 
                    for data in historical_themes.values()
                ),
                'themes': [
                    {
                        'name': theme,
                        'confidence': data.confidence_score,
                        'concepts': dict(data.concept_weights)
                    }
                    for theme, data in historical_themes.items()
                ]
            }
            temporal_context.append(context_entry)
        
        return temporal_context

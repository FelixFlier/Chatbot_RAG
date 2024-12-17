# model_classes.py
from dataclasses import dataclass, field
from typing import Dict, List, Set, Optional, Tuple, Any
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

class IntelligentIntentRecognizer:
    def __init__(self, vectorstore):
        self.vectorstore = vectorstore
        self.nlp = spacy.load("de_core_news_md")
        self.pattern_cache = {}
        self.intent_history = []
        
        self.intent_categories = {
            "definition": {
                "patterns": ["was ist", "definiere", "bedeutet", "erkläre"],
                "response_type": "explanation"
            },
            "process": {
                "patterns": ["wie", "workflow", "prozess", "ablauf"],
                "response_type": "step_by_step"
            },
            "comparison": {
                "patterns": ["vergleich", "unterschied", "versus"],
                "response_type": "comparison"
            },
            "analysis": {
                "patterns": ["analysiere", "bewerte", "untersuche"],
                "response_type": "analysis"
            },
            "application": {
                "patterns": ["anwenden", "implementieren", "umsetzen"],
                "response_type": "instruction"
            },
            "context": {
                "patterns": ["zusammenhang", "kontext", "bezug"],
                "response_type": "contextual"
            }
        }
    

    def analyze_intent(self, query: str, context: Optional[dict] = None) -> dict:
        # Erweitere die Intent-Patterns
        follow_up_patterns = {
            'how': ['wie', 'wodurch', 'auf welche weise'],
            'why': ['warum', 'weshalb', 'wieso'],
            'what': ['was bedeutet', 'was ist', 'was sind'],
            'when': ['wann', 'zu welchem zeitpunkt'],
            'where': ['wo', 'an welchem ort']
        }
        
        # Basis Intent-Kategorien aus der ursprünglichen Implementierung
        base_intent = self._pattern_analysis(query)
        
        # Prüfe auf Follow-up-Muster
        query_lower = query.lower()
        for intent_type, patterns in follow_up_patterns.items():
            if any(pattern in query_lower for pattern in patterns):
                # Verknüpfe mit vorherigem Kontext
                if context and context.get('last_intent'):
                    return {
                        'intent': context['last_intent'],
                        'is_followup': True,
                        'followup_type': intent_type,
                        'confidence': 0.8,
                        'response_type': self.intent_categories[context['last_intent']]['response_type']
                    }
        
        # Wenn kein Follow-up erkannt wurde, gib den Basis-Intent zurück
        return {
            'intent': base_intent['intent'],
            'is_followup': False,
            'confidence': base_intent['confidence'],
            'response_type': base_intent['response_type']
        }

    # Die ursprüngliche _pattern_analysis Methode bleibt als Fallback
    def _pattern_analysis(self, query: str) -> dict:
        query_lower = query.lower()
        max_confidence = 0.0
        detected_intent = None
        
        for intent, data in self.intent_categories.items():
            confidence = sum(1 for pattern in data['patterns'] if pattern in query_lower)
            if confidence > max_confidence:
                max_confidence = confidence
                detected_intent = intent
        
        if not detected_intent:
            detected_intent = "definition"  # Fallback Intent
            max_confidence = 0.5
            
        return {
            'intent': detected_intent,
            'confidence': max_confidence / (max_confidence + 1) if max_confidence > 0 else 0.5,
            'response_type': self.intent_categories[detected_intent]['response_type'],
            'method': 'pattern'
        }

    def _context_analysis(self, query: str, context: Optional[dict]) -> dict:
        """Vereinfachte Kontextanalyse"""
        if not context or not self.intent_history:
            return {'confidence': 0.0, 'method': 'context', 'intent': 'definition'}
            
        # Nutze den letzten Intent als Kontext
        last_intent = self.intent_history[-1]
        return {
            'intent': last_intent['intent'],
            'confidence': 0.4,  # Reduzierte Konfidenz für Kontext-basierte Entscheidungen
            'method': 'context'
        }

    def _update_intent_history(self, intent: dict, query: str):
        self.intent_history.append({
            'intent': intent['intent'],
            'query': query,
            'timestamp': datetime.now()
        })
        
        if len(self.intent_history) > 10:
            self.intent_history.pop(0)

class DynamicConceptAnalyzer:
    """Vereinfachte Konzeptanalyse"""
    
    def __init__(self, vectorstore):
        self.vectorstore = vectorstore
        self.embeddings_model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
        )
        self.nlp = spacy.load("de_core_news_md")
        self.concept_cache = {}
        self.entity_cache = {}
    
    def analyze_query_concepts(self, query: str, context_window: int = 3) -> Dict[str, Any]:
        if query in self.concept_cache:
            return self.concept_cache[query]
        
        doc = self.nlp(query)
        
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

class EnhancedHybridSearcher:
    def __init__(self, vectorstore, bm25, thematic_tracker):
        self.vectorstore = vectorstore
        self.bm25 = bm25
        self.thematic_tracker = thematic_tracker
        self.chunk_graph = self._build_chunk_graph()
        self.search_history = []
        self.minimum_similarity_threshold = 0.6

    def _build_chunk_graph(self) -> nx.DiGraph:
        """Erstellt Graph-Struktur für Chunk-Beziehungen"""
        graph = nx.DiGraph()
        try:
            # Korrekte Extraktion der Dokumente aus FAISS
            docs = []
            if hasattr(self.vectorstore, 'docstore'):
                # Korrekter Zugriff auf InMemoryDocstore
                for doc_id in self.vectorstore.index_to_docstore_id:
                    # InMemoryDocstore verwendet dict-ähnlichen Zugriff
                    doc = self.vectorstore.docstore.search(doc_id)
                    if isinstance(doc, Document):
                        docs.append(doc)
                    elif isinstance(doc, (str, bytes)):
                        # Konvertiere String zu Document
                        docs.append(Document(page_content=str(doc), metadata={}))
            
            if not docs:
                print("Warnung: Keine Dokumente gefunden")
                return graph
            
            # Füge Knoten hinzu
            for i, doc in enumerate(docs):
                graph.add_node(
                    i,
                    content=doc.page_content,
                    metadata=getattr(doc, 'metadata', {})
                )
            
            return graph
            
        except Exception as e:
            print(f"Fehler beim Erstellen des Chunk-Graphen: {str(e)}")
            return graph

    def _add_graph_edges(self, graph: nx.DiGraph, docs: List[Document]):
        """Fügt Kanten zwischen ähnlichen Dokumenten hinzu"""
        try:
            for i, doc1 in enumerate(docs):
                # Berechne Embedding für doc1
                if hasattr(self.vectorstore, 'embedding_function'):
                    embedding1 = self.vectorstore.embedding_function.embed_query(doc1.page_content)
                    
                    for j, doc2 in enumerate(docs[i+1:], i+1):
                        embedding2 = self.vectorstore.embedding_function.embed_query(doc2.page_content)
                        similarity = np.dot(embedding1, embedding2) / (
                            np.linalg.norm(embedding1) * np.linalg.norm(embedding2)
                        )
                        
                        if similarity > self.minimum_similarity_threshold:
                            graph.add_edge(i, j, weight=float(similarity))
        except Exception as e:
            print(f"Fehler beim Hinzufügen von Kanten: {str(e)}")

    def hybrid_search(self, query: str, context: Optional[Dict[str, Any]] = None, k: int = 5) -> List[Dict[str, Any]]:
        """Führt die hybride Suche durch"""
        try:
            # Semantic Search
            semantic_results = self._semantic_search(query, k)
            
            # Keyword Search
            keyword_results = self._keyword_search(query, k)
            
            # Kombiniere Ergebnisse
            combined_results = self._combine_results(semantic_results, keyword_results)
            
            # Erweitere mit Kontext
            enhanced_results = self._enhance_with_context(combined_results, query)
            
            # Aktualisiere Suchhistorie
            self._update_search_history(query, enhanced_results)
            
            return enhanced_results[:k]
            
        except Exception as e:
            print(f"Fehler in hybrid_search: {str(e)}")
            return []

    def _semantic_search(self, query: str, k: int) -> List[Dict[str, Any]]:
        """Führt semantische Suche durch"""
        try:
            results = self.vectorstore.similarity_search_with_score(query, k=k)
            return [
                {
                    'content': doc.page_content if isinstance(doc, Document) else str(doc),
                    'score': float(score),
                    'type': 'semantic',
                    'metadata': getattr(doc, 'metadata', {}) if isinstance(doc, Document) else {}
                }
                for doc, score in results
            ]
        except Exception as e:
            print(f"Fehler in semantic search: {str(e)}")
            return []

    def _keyword_search(self, query: str, k: int) -> List[Dict[str, Any]]:
        """Führt Keyword-Suche durch"""
        try:
            query_tokens = query.lower().split()
            scores = self.bm25.get_scores(query_tokens)
            
            # Hole das Dokument-Corpus
            corpus = []
            if hasattr(self.bm25, '_corpus'):
                corpus = self.bm25._corpus
            elif hasattr(self.bm25, 'corpus'):
                corpus = self.bm25.corpus
            elif hasattr(self.vectorstore, 'docstore'):
                # Korrekter Zugriff auf FAISS Dokumente
                corpus = []
                for doc_id in self.vectorstore.index_to_docstore_id:
                    doc = self.vectorstore.docstore.search(doc_id)
                    if isinstance(doc, Document):
                        corpus.append(doc.page_content.split())
                    elif isinstance(doc, (str, bytes)):
                        corpus.append(str(doc).split())
            
            if not corpus:
                return []
            
            scored_docs = list(enumerate(scores))
            scored_docs.sort(key=lambda x: x[1], reverse=True)
            
            results = []
            for idx, score in scored_docs[:k]:
                if score > 0 and idx < len(corpus):
                    doc_content = " ".join(corpus[idx])
                    results.append({
                        'content': doc_content,
                        'score': float(score),
                        'type': 'keyword',
                        'metadata': {'index': idx}
                    })
            
            return results
            
        except Exception as e:
            print(f"Fehler in keyword search: {str(e)}")
            return []

    def _combine_results(self, semantic_results: List[Dict], keyword_results: List[Dict]) -> List[Dict]:
        """Kombiniert die Suchergebnisse"""
        combined_scores = {}
        weights = {'semantic': 0.6, 'keyword': 0.4}
        
        for result in semantic_results:
            combined_scores[result['content']] = {
                'score': result['score'] * weights['semantic'],
                'sources': ['semantic'],
                'metadata': result['metadata']
            }

        for result in keyword_results:
            if result['content'] in combined_scores:
                combined_scores[result['content']]['score'] += (
                    result['score'] * weights['keyword']
                )
                combined_scores[result['content']]['sources'].append('keyword')
            else:
                combined_scores[result['content']] = {
                    'score': result['score'] * weights['keyword'],
                    'sources': ['keyword'],
                    'metadata': result['metadata']
                }

        return sorted(
            [
                {
                    'content': content,
                    'score': data['score'],
                    'sources': data['sources'],
                    'metadata': data['metadata']
                }
                for content, data in combined_scores.items()
            ],
            key=lambda x: x['score'],
            reverse=True
        )

    def _enhance_with_context(self, results: List[Dict], query: str) -> List[Dict]:
        """Erweitert Ergebnisse mit Kontext und Qualitätsmetriken"""
        for result in results:
            try:
                # Berechne Qualitätsmetriken
                quality_metrics = {
                    'information_density': self._calculate_information_density(result['content']),
                    'relevance_score': result['score']
                }
                
                # Füge Metriken hinzu
                result['quality_metrics'] = quality_metrics
                
            except Exception as e:
                print(f"Fehler bei Kontexterweiterung: {str(e)}")
                result['quality_metrics'] = {
                    'information_density': 0.5,
                    'relevance_score': result['score']
                }
                
        return results

    def _calculate_information_density(self, content: str) -> float:
        """Berechnet die Informationsdichte eines Textes"""
        try:
            words = content.split()
            unique_words = set(words)
            return len(unique_words) / max(len(words), 1)
        except Exception as e:
            print(f"Fehler bei Dichteberechnung: {str(e)}")
            return 0.5

    def _update_search_history(self, query: str, results: List[Dict]):
        """Aktualisiert die Suchhistorie"""
        try:
            self.search_history.append({
                'query': query,
                'num_results': len(results),
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'average_quality': np.mean([
                    r.get('quality_metrics', {}).get('information_density', 0.5)
                    for r in results
                ]) if results else 0.0
            })
            
            if len(self.search_history) > 100:
                self.search_history = self.search_history[-100:]
                
        except Exception as e:
            print(f"Fehler bei Historien-Update: {str(e)}")

class ResponseGenerator:
    def __init__(self, llm, embeddings_model):
        self.llm = llm
        self.embeddings_model = embeddings_model
        self.response_templates = self._initialize_templates()
        self.quality_checker = ResponseQualityChecker(embeddings_model)
        self.intent_introductions = {
            "definition": "Lassen Sie mich Ihnen eine präzise Definition des angefragten Konzepts geben:",
            "process": "Ich erläutere Ihnen den gewünschten Prozess Schritt für Schritt:",
            "comparison": "Im Folgenden stelle ich einen strukturierten Vergleich der genannten Aspekte an:",
            "analysis": "Hier ist meine detaillierte Analyse zu Ihrer Anfrage:",
            "application": "Ich erkläre Ihnen die praktische Anwendung wie folgt:",
            "context": "Lassen Sie mich den relevanten Kontext für Sie aufschlüsseln:"
        }

    def _initialize_templates(self) -> Dict[str, PromptTemplate]:
        """Initialisiert kompaktere Antwort-Templates"""
        return {
            "definition": PromptTemplate(
                input_variables=["context", "question"],
                template="""
                Kontext: {context}
                Frage: {question}
                
                Generiere eine präzise, fokussierte Antwort mit folgenden Punkten:
                - Klare Definition des Konzepts
                - Wichtigste Kernaspekte
                - Ein kurzes Beispiel falls hilfreich
                
                Halte die Antwort kompakt und relevant zum Kontext.
                
                Antwort:"""
            ),
            "process": PromptTemplate(
                input_variables=["context", "question"],
                template="""
                Kontext: {context}
                Frage: {question}
                
                Beschreibe den Prozess knapp und präzise:
                - Kurze Prozessübersicht
                - Wesentliche Schritte
                - Kritische Punkte
                
                Fokussiere auf die wichtigsten Informationen aus dem Kontext.
                
                Antwort:"""
            ),
            "analysis": PromptTemplate(
                input_variables=["context", "question"],
                template="""
                Kontext: {context}
                Frage: {question}
                
                Erstelle eine fokussierte Analyse:
                - Kernpunkte der Situation
                - Wichtigste Erkenntnisse
                - Zentrale Schlussfolgerung
                
                Bleibe präzise und relevant zum gegebenen Kontext.
                
                Antwort:"""
            ),
            "comparison": PromptTemplate(
                input_variables=["context", "question"],
                template="""
                Kontext: {context}
                Frage: {question}
                
                Stelle einen knappen, strukturierten Vergleich an:
                - Hauptunterschiede
                - Gemeinsamkeiten
                - Praktische Bedeutung
                
                Konzentriere dich auf die relevantesten Aspekte aus dem Kontext.
                
                Antwort:"""
            )
        }

    def generate_response(self, query: str, context: List[Dict], intent_info: Dict) -> Dict[str, Any]:
        """Generiert eine qualitätsgeprüfte Antwort"""
        try:
            # Template Auswahl
            template = self.response_templates.get(
                intent_info['intent'],
                self.response_templates['definition']
            )

            # LLMChain erstellen
            chain = LLMChain(
                llm=self.llm,
                prompt=template
            )

            # Antwort generieren
            response = chain.invoke({
                "context": self._prepare_context(context),
                "question": query
            })

            # Antwort extrahieren und bereinigen
            response_text = self._clean_response(
                response.get('text', '') if isinstance(response, dict) else str(response)
            )

            # Qualitätsprüfung
            quality_check = self.quality_checker.check_response(
                response_text,
                query,
                context,
                intent_info
            )
            
            # Prüfe Qualitätsschwellen
            failed_metrics = []
            quality_thresholds = {
                'relevance_score': 0.35,
                'completeness_score': 0.001,
                'coherence_score': 0.35,
                'accuracy_score': 0.35
            }
            
            for metric, threshold in quality_thresholds.items():
                if quality_check.get('metrics', {}).get(metric, 0) < threshold:
                    failed_metrics.append(metric)
            
            if failed_metrics:
                return {
                    'answer': ("Entschuldigung, ich konnte keine qualitativ ausreichende Antwort "
                            "generieren. Bitte formulieren Sie Ihre Frage anders oder geben Sie "
                            "mehr Kontext."),
                    'metadata': {
                        'intent': intent_info,
                        'quality_metrics': quality_check,
                        'failed_quality_check': True,
                        'failed_metrics': failed_metrics,
                        'context_coverage': 0.0
                    }
                }
            
            # Erfolgreiche Antwort formatieren
            introduction = self.intent_introductions.get(
                intent_info['intent'],
                "Hier ist meine Antwort auf Ihre Frage:"
            )
            final_response = f"{introduction}\n\n{response_text}\n\nHaben Sie noch weitere Fragen zu diesem Thema?"
            
            # Berechne Kontext-Coverage
            context_coverage = self._calculate_context_coverage(response_text, context)
            
            # Erstelle erweiterte Qualitätsmetriken
            enhanced_metrics = {
                **quality_check.get('metrics', {}),
                'context_relevance': context[0].get('quality_metrics', {}).get('relevance_score', 0.5) if context else 0.5,
                'information_density': context[0].get('quality_metrics', {}).get('information_density', 0.5) if context else 0.5
            }
            
            return {
                'answer': final_response,
                'metadata': {
                    'intent': intent_info,
                    'quality_metrics': {
                        'metrics': enhanced_metrics,
                        'overall_quality': np.mean(list(enhanced_metrics.values())),
                        'threshold_check': 'passed'
                    },
                    'context_coverage': context_coverage,
                    'failed_quality_check': False,
                    'failed_metrics': [],
                    'sources': [
                        {
                            'content': ctx.get('content', ''),
                            'metadata': ctx.get('metadata', {}),
                            'score': ctx.get('score', 0.0)
                        }
                        for ctx in context[:3]  # Top 3 Quellen
                    ]
                }
            }
                
        except Exception as e:
            print(f"Error in generate_response: {str(e)}")
            return {
                'answer': ("Es tut mir leid, aber ich konnte keine angemessene Antwort generieren. "
                        "Können Sie Ihre Frage bitte anders formulieren?"),
                'metadata': {
                    'error': str(e),
                    'failed_quality_check': True,
                    'failed_metrics': ['processing_error'],
                    'context_coverage': 0.0,
                    'quality_metrics': {
                        'metrics': {},
                        'overall_quality': 0.0,
                        'threshold_check': 'failed'
                    }
                }
            }
    def _prepare_context(self, context: List[Dict]) -> str:
        if not context:
            return "Keine relevanten Informationen verfügbar."
        
        sorted_context = sorted(context, key=lambda x: x['score'], reverse=True)
        combined_text = []
        total_length = 0
        max_length = 1000  # Reduziert von 2000 für kompaktere Antworten

        for ctx in sorted_context:
            if total_length + len(ctx['content']) > max_length:
                break
            combined_text.append(ctx['content'])
            total_length += len(ctx['content'])

        return " ".join(combined_text)

    def _clean_response(self, text: str) -> str:
        if "Antwort:" in text:
            text = text.split("Antwort:")[-1].strip()
        
        markers = [
            'Kontext:', 
            'Frage:', 
            'Beschreibe den Prozess',
            'Liefere eine direkte Analyse',
            'Stelle einen direkten Vergleich an',
            'Beginne direkt mit der Erklärung',
            'ohne Einleitung',
            'oder Wiederholung der Frage'
        ]
        
        for marker in markers:
            text = text.replace(marker, '').strip()
        
        text = re.sub(r'```.*?```', '', text, flags=re.DOTALL)
        text = re.sub(r'`.*?`', '', text)
        text = re.sub(r'\*\*.*?\*\*', '', text)
        text = re.sub(r'\n\s*\n', '\n\n', text)
        
        return text.strip()

    def _calculate_context_coverage(self, response: str, context: List[Dict]) -> float:
        try:
            response_embedding = self.embeddings_model.embed_query(response)
            context_embeddings = [
                self.embeddings_model.embed_query(ctx['content'])
                for ctx in context
            ]

            similarities = [
                np.dot(response_embedding, ctx_emb) / (
                    np.linalg.norm(response_embedding) * np.linalg.norm(ctx_emb)
                )
                for ctx_emb in context_embeddings
            ]

            return float(np.mean(similarities))
        except Exception as e:
            print(f"Error in context coverage calculation: {str(e)}")
            return 0.5


class ResponseQualityChecker:
    def __init__(self, embeddings_model):
        self.embeddings_model = embeddings_model
        self.quality_thresholds = {
            'relevance': 0.7,
            'completeness': 0.7,
            'coherence': 0.7,
            'accuracy': 0.7
        }
        self._similarity_cache = {}

    def check_response(self, response: str, query: str, context: List[Dict], intent_info: Dict) -> Dict[str, Any]:
    # Erweiterte Qualitätsprüfung
        if intent_info.get('is_followup'):
            # Strengere Bewertung für Follow-up-Fragen
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
    
    def __init__(self, embeddings_model):
        self.embeddings_model = embeddings_model
        self.current_themes: Dict[str, EnhancedThemeMetadata] = {}
        self.theme_history: List[Dict[str, EnhancedThemeMetadata]] = []
        self.decay_factor = 0.9
        self.max_history_length = 10
        self.context_window = 5
        self.theme_relations = defaultdict(list)
        
    def update_themes(self, message: str, detected_themes: List[str], 
                     concepts: Dict[str, Any], intent_info: Dict[str, Any],
                     context: Optional[Dict] = None) -> Dict[str, Any]:
        """Aktualisiert Themen mit erweiterter Kontextanalyse"""
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
            if theme in self.current_themes:
                theme_data = self.current_themes[theme]
                theme_data.mention_count += 1
                theme_data.last_mentioned = current_time
                theme_data.confidence_score = min(
                    1.0, 
                    theme_data.confidence_score + 0.1
                )
                updates['updated_themes'].append(theme)
            else:
                self.current_themes[theme] = EnhancedThemeMetadata(
                    name=theme,
                    last_mentioned=current_time
                )
                updates['new_themes'].append(theme)
            
            # Update theme data
            theme_data = self.current_themes[theme]
            
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
            
            # Prüfe auf Themenverschiebungen
            if self._detect_theme_shift(theme, message):
                updates['theme_shifts'].append(theme)
        
        self._archive_old_themes(current_time)
        self._update_theme_relations()
        
        return updates
    
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
    
    def _detect_theme_shift(self, theme: str, message: str) -> bool:
        """Erkennt Themenwechsel mit verbesserter Analyse"""
        if theme not in self.current_themes:
            return True
            
        theme_data = self.current_themes[theme]
        
        # Prüfe Ähnlichkeit mit aktuellem Kontext
        if theme_data.context_history:
            message_embedding = self.embeddings_model.embed_query(message)
            context_embedding = self.embeddings_model.embed_query(
                theme_data.context_history[-1]['context'].get('text', '')
            )
            
            similarity = np.dot(message_embedding, context_embedding) / (
                np.linalg.norm(message_embedding) * 
                np.linalg.norm(context_embedding)
            )
            
            return similarity < 0.3
            
        return True
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
    
    def _update_theme_relations(self):
        """Aktualisiert Beziehungen zwischen Themen"""
        for theme1 in self.current_themes:
            theme1_data = self.current_themes[theme1]
            for theme2 in self.current_themes:
                if theme1 != theme2:
                    theme2_data = self.current_themes[theme2]
                    
                    # Berechne Ähnlichkeit der Konzepte
                    similarity = self._calculate_theme_similarity(
                        theme1_data,
                        theme2_data
                    )
                    
                    if similarity > 0.3:
                        self.theme_relations[theme1].append({
                            'target': theme2,
                            'similarity': similarity,
                            'timestamp': datetime.now()
                        })
    
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
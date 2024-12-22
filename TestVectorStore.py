import unittest
import numpy as np
import os
import tempfile
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from rank_bm25 import BM25Okapi
import pickle
import time
from typing import List, Dict, Any, Tuple
import logging
import torch
from collections import defaultdict
from functools import lru_cache
import spacy

# Pfade definieren
TEXT_PATH = r"C:\Users\felix\Downloads\Machine Learning\text.txt"
SAVE_PATH = r"C:\Users\felix\LangchainProject\vector_stores"
LOG_PATH = os.path.join(SAVE_PATH, 'vectorstore.log')

# Erstelle Speicherordner falls nicht vorhanden
os.makedirs(SAVE_PATH, exist_ok=True)

# Logging konfigurieren
logging.basicConfig(
    filename=LOG_PATH,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

import os
import numpy as np
import logging
import pickle
import time
import spacy
import torch
from typing import List, Dict, Any, Tuple, Set, Optional
from dataclasses import dataclass, field
from datetime import datetime
from collections import defaultdict
from functools import lru_cache
from pathlib import Path

from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from rank_bm25 import BM25Okapi
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans

# Logging Setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class HierarchyInfo:
    """Speichert Informationen über die hierarchische Position eines Chunks"""
    level: int
    section: Optional[str] = None
    subsection: Optional[str] = None
    bullet_points: List[str] = field(default_factory=list)

class EnhancedVectorStoreManager:
    def __init__(self, base_path: str):
        """
        Initialisiert den Vector Store Manager mit optimierten Einstellungen.
        
        Args:
            base_path: Pfad zum Speichern der Vector Stores
        """
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        
        # Initialisiere Komponenten
        self.text_splitter = self._initialize_text_splitter()
        self.embeddings = self._initialize_embeddings()
        self.nlp = self._initialize_spacy()
        
        # Zwischenspeicher
        self.section_structure = {}
        self.keyword_index = defaultdict(list)
        self.entity_index = defaultdict(list)
        
        # Performance Tracking
        self.processing_metrics = defaultdict(list)

    def _initialize_text_splitter(self) -> RecursiveCharacterTextSplitter:
        """Initialisiert optimierten TextSplitter für hierarchische Dokumentstruktur"""
        return RecursiveCharacterTextSplitter(
            chunk_size=600,  # Optimale Größe für Unterkapitel
            chunk_overlap=100,
            separators=[
                "\n# ",     # Hauptthemen
                "\n## ",    # Unterkapitel
                "\n- ",     # Aufzählungspunkte
                "\n",       # Zeilenumbrüche
                ". ",       # Sätze
                ", ",       # Aufzählungen innerhalb von Sätzen
                " "         # Wörter als letzte Option
            ],
            length_function=len,
        )

    def _initialize_embeddings(self) -> HuggingFaceEmbeddings:
        """Initialisiert das Embedding-Modell mit GPU-Unterstützung wenn verfügbar"""
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logger.info(f"Verwende Device: {device}")
        
        return HuggingFaceEmbeddings(
            model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
            model_kwargs={'device': device}
        )

    def _initialize_spacy(self):
        """Initialisiert das Spacy-Modell"""
        try:
            return spacy.load("de_core_news_sm")
        except OSError:
            logger.info("Lade Spacy-Modell nach...")
            os.system("python -m spacy download de_core_news_sm")
            return spacy.load("de_core_news_sm")

    def process_text(self, text: str) -> Tuple[List[Document], Dict[str, Any]]:
        """
        Verarbeitet Text mit Berücksichtigung der hierarchischen Struktur.
        
        Args:
            text: Einzulesender Text
            
        Returns:
            Tuple aus Liste von Dokumenten und Verarbeitungsmetriken
        """
        start_time = time.time()
        try:
            # Parse Dokumentstruktur
            self.section_structure = self._parse_document_structure(text)
            chunks = self.text_splitter.split_text(text)
            
            documents = []
            metrics = {
                'total_chunks': len(chunks),
                'sections': len(self.section_structure),
                'hierarchy_levels': defaultdict(int),
                'processing_time': 0,
                'avg_chunk_size': sum(len(chunk) for chunk in chunks) / len(chunks)
            }
            
            # Verarbeite Chunks parallel wenn möglich
            for i, chunk in enumerate(chunks):
                # Bestimme hierarchische Position
                hierarchy_info = self._get_hierarchy_info(chunk)
                metrics['hierarchy_levels'][hierarchy_info.level] += 1
                
                # Verarbeite Chunk
                doc = self._process_chunk(chunk, i, hierarchy_info)
                documents.append(doc)
                
                # Status-Update
                if (i + 1) % 100 == 0:
                    logger.info(f"Verarbeitet: {i+1}/{len(chunks)} Chunks")
            
            # Berechne finale Metriken
            metrics['processing_time'] = time.time() - start_time
            
            return documents, metrics
            
        except Exception as e:
            logger.error(f"Fehler bei der Textverarbeitung: {str(e)}")
            raise

    def _process_chunk(self, chunk: str, chunk_id: int, hierarchy_info: HierarchyInfo) -> Document:
        """Verarbeitet einen einzelnen Chunk"""
        # NLP-Analyse
        doc_nlp = self.nlp(chunk)
        
        # Extrahiere Features
        keywords = self._extract_keywords(doc_nlp)
        entities = self._extract_entities(doc_nlp)
        embedding = self._embed_text(chunk)
        importance = self._calculate_importance(chunk, hierarchy_info, keywords)
        
        # Erstelle Document-Objekt mit optimierten Metadaten
        return Document(
            page_content=chunk,
            metadata={
                "chunk_id": f"chunk_{chunk_id}",
                "hierarchy_level": hierarchy_info.level,
                "section": hierarchy_info.section,
                "subsection": hierarchy_info.subsection,
                "is_bullet_point": chunk.strip().startswith("-"),
                "keywords": keywords,
                "entities": entities,
                "embedding": embedding,
                "importance_score": importance,
                "neighboring_chunks": {
                    "prev": chunk_id-1 if chunk_id > 0 else None,
                    "next": chunk_id+1 if chunk_id < self._get_total_chunks()-1 else None
                },
                "term_frequency": self._calculate_term_frequency(doc_nlp),
                "semantic_density": len(set(token.text.lower() for token in doc_nlp)) / len(doc_nlp),
                "created_at": datetime.now().isoformat()
            }
        )

    def _parse_document_structure(self, text: str) -> Dict[str, Dict[str, List[str]]]:
        """Extrahiert die hierarchische Dokumentstruktur mit verbesserter Fehlerbehandlung"""
        sections = {}
        current_main = ""
        current_sub = ""
        
        try:
            for line in text.split('\n'):
                line = line.strip()
                if not line:  # Überspringe leere Zeilen
                    continue
                    
                if line.startswith('# '):
                    current_main = line[2:].strip()
                    sections[current_main] = {}
                    current_sub = ""  # Reset current_sub bei neuer Hauptsektion
                    
                elif line.startswith('## '):
                    if not current_main:  # Falls keine Hauptsektion definiert
                        logger.warning(f"Untersektion ohne Hauptsektion gefunden: {line}")
                        current_main = "Unzugeordnete Themen"
                        sections[current_main] = {}
                        
                    current_sub = line[3:].strip()
                    sections[current_main][current_sub] = []
                    
                elif line.startswith('- '):
                    # Sicherheitscheck und Fallback
                    if not current_main:
                        logger.warning("Aufzählungspunkt ohne Hauptsektion gefunden")
                        current_main = "Unzugeordnete Themen"
                        sections[current_main] = {}
                    
                    if not current_sub:
                        logger.warning("Aufzählungspunkt ohne Untersektion gefunden")
                        current_sub = "Allgemeine Punkte"
                        sections[current_main][current_sub] = []
                    
                    if current_main not in sections:
                        sections[current_main] = {}
                    if current_sub not in sections[current_main]:
                        sections[current_main][current_sub] = []
                        
                    sections[current_main][current_sub].append(line[2:].strip())
            
            logger.info(f"Dokumentstruktur erfolgreich geparst: {len(sections)} Hauptsektionen")
            return sections
            
        except Exception as e:
            logger.error(f"Fehler beim Parsen der Dokumentstruktur: {str(e)}")
            raise

    def _get_hierarchy_info(self, chunk: str) -> HierarchyInfo:
        """Bestimmt die hierarchische Position eines Chunks"""
        chunk = chunk.strip()
        
        # Suche nach Hauptthema
        for main_section, subsections in self.section_structure.items():
            if main_section in chunk:
                # Suche nach Unterkapitel
                for subsection, bullets in subsections.items():
                    if subsection in chunk:
                        return HierarchyInfo(
                            level=2,
                            section=main_section,
                            subsection=subsection,
                            bullet_points=[b for b in bullets if b in chunk]
                        )
                return HierarchyInfo(level=1, section=main_section)
                
        # Wenn keine spezifische Position gefunden wurde
        return HierarchyInfo(level=3)

    def create_and_save_stores(self, documents: List[Document]) -> Dict[str, Any]:
        """Erstellt und speichert optimierte Vector Stores"""
        try:
            logger.info("Erstelle optimierte Vector Stores...")
            
            # FAISS Vector Store mit optimierten Einstellungen
            vectorstore = FAISS.from_documents(
                documents,
                self.embeddings,
                distance_strategy="cos"  # Optimiert für Semantische Suche
            )
            
            # Optimierte BM25 Vorbereitung
            tokenized_chunks = []
            section_index = defaultdict(list)
            keyword_index = defaultdict(list)
            
            for i, doc in enumerate(documents):
                # Tokenisierung für BM25
                tokens = self._preprocess_text(doc.page_content)
                tokenized_chunks.append(tokens)
                
                # Indizes aufbauen
                if doc.metadata['section']:
                    section_index[doc.metadata['section']].append(i)
                
                for keyword in doc.metadata['keywords']:
                    keyword_index[keyword].append(i)
            
            # Erstelle BM25
            bm25 = BM25Okapi(tokenized_chunks)
            
            # Speichere zusätzliche Suchstrukturen
            stores = {
                'vectorstore': vectorstore,
                'bm25': bm25,
                'documents': documents,
                'section_index': dict(section_index),
                'keyword_index': dict(keyword_index),
                'section_structure': self.section_structure,
                'metadata': {
                    'created_at': datetime.now().isoformat(),
                    'num_documents': len(documents),
                    'sections': list(self.section_structure.keys())
                }
            }
            
            # Speichere alle Stores
            for name, store in stores.items():
                path = self.base_path / f'{name}.pkl'
                with open(path, 'wb') as f:
                    pickle.dump(store, f, protocol=pickle.HIGHEST_PROTOCOL)
                logger.info(f"Gespeichert: {path}")
            
            return stores
            
        except Exception as e:
            logger.error(f"Fehler beim Speichern der Stores: {str(e)}")
            raise

    @lru_cache(maxsize=1000)
    def _embed_text(self, text: str) -> np.ndarray:
        """Cached Embedding-Generierung"""
        return self.embeddings.embed_query(text)

    def _extract_keywords(self, doc_nlp) -> List[str]:
        """Extrahiert relevante Keywords"""
        return [
            token.text.lower()
            for token in doc_nlp
            if not token.is_stop and not token.is_punct and token.pos_ in {'NOUN', 'VERB', 'ADJ'}
        ]

    def _extract_entities(self, doc_nlp) -> List[Dict[str, str]]:
        """Extrahiert benannte Entitäten"""
        return [
            {
                'text': ent.text,
                'label': ent.label_,
                'description': spacy.explain(ent.label_)
            }
            for ent in doc_nlp.ents
        ]

    def _calculate_importance(
        self,
        chunk: str,
        hierarchy_info: HierarchyInfo,
        keywords: List[str]
    ) -> float:
        """Berechnet den Wichtigkeits-Score eines Chunks"""
        # Basis-Score nach Hierarchie-Level
        base_score = {1: 1.0, 2: 0.8, 3: 0.6}.get(hierarchy_info.level, 0.5)
        
        # Keyword-Dichte
        keyword_density = len(keywords) / len(chunk.split())
        
        # Bullet-Point Bonus
        bullet_bonus = 0.1 if hierarchy_info.bullet_points else 0
        
        # Kombiniere Scores
        importance = (base_score * 0.5 + keyword_density * 0.3 + bullet_bonus * 0.2)
        
        return min(max(importance, 0.1), 1.0)

    def _calculate_term_frequency(self, doc_nlp) -> Dict[str, int]:
        """Berechnet Termhäufigkeiten"""
        term_freq = defaultdict(int)
        for token in doc_nlp:
            if not token.is_stop and not token.is_punct:
                term_freq[token.text.lower()] += 1
        return dict(term_freq)

    def _preprocess_text(self, text: str) -> List[str]:
        """Vorverarbeitung für BM25"""
        doc = self.nlp(text.lower())
        return [
            token.lemma_
            for token in doc
            if not token.is_stop and not token.is_punct and token.is_alpha
        ]

    def _get_total_chunks(self) -> int:
        """Hilfsmethod zum Zählen der Chunks"""
        return sum(
            1 + len(subsections) + sum(len(bullets) for bullets in subsections.values())
            for subsections in self.section_structure.values()
        )

    def cleanup(self):
        """Räumt temporäre Ressourcen auf"""
        self.keyword_index.clear()
        self.entity_index.clear()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    def print_metrics(self, metrics: Dict[str, Any]):
        """Druckt formatierte Verarbeitungsmetriken"""
        print("\nVerarbeitungsmetriken:")
        print(f"Anzahl Chunks: {metrics['total_chunks']}")
        print(f"Anzahl Sektionen: {metrics['sections']}")
        print(f"Durchschnittliche Chunk-Größe: {metrics['avg_chunk_size']:.2f} Zeichen")
        print("\nHierarchie-Level Verteilung:")
        for level, count in metrics['hierarchy_levels'].items():
            print(f"Level {level}: {count} Chunks")
        print(f"\nVerarbeitungszeit: {metrics['processing_time']:.2f} Sekunden")

    def validate_stores(self, stores: Dict[str, Any]):
        """Validiert die erstellten Stores"""
        required_stores = {'vectorstore', 'bm25', 'documents', 'section_index', 
                        'keyword_index', 'section_structure', 'metadata'}
        
        missing_stores = required_stores - set(stores.keys())
        if missing_stores:
            raise ValueError(f"Fehlende Stores: {missing_stores}")

        # Validiere Dokumente
        if not stores['documents']:
            raise ValueError("Keine Dokumente im Store")

        # Validiere Indizes
        if not stores['section_index']:
            logger.warning("Sektions-Index ist leer")
        if not stores['keyword_index']:
            logger.warning("Keyword-Index ist leer")

    def test_search(self, stores: Dict[str, Any], query: str = "IFRS Standards"):
        """Führt eine Test-Suche durch"""
        print(f"\nFühre Test-Suche durch: '{query}'")
        
        # Vector Search
        vector_results = stores['vectorstore'].similarity_search_with_score(query, k=3)
        
        # BM25 Search
        tokens = self._preprocess_text(query)
        bm25_scores = stores['bm25'].get_scores(tokens)
        top_bm25 = sorted(enumerate(bm25_scores), key=lambda x: x[1], reverse=True)[:3]
        
        print("\nTop 3 Vector Search Ergebnisse:")
        for doc, score in vector_results:
            print(f"\nScore: {1-score:.3f}")
            print(f"Text: {doc.page_content[:200]}...")

        print("\nTop 3 BM25 Ergebnisse:")
        for idx, score in top_bm25:
            print(f"\nScore: {score:.3f}")
            print(f"Text: {stores['documents'][idx].page_content[:200]}...")

def main():
    """Hauptfunktion"""
    # Konfiguration
    base_path = Path("vector_stores")
    text_path = Path("text.txt")
    
    try:
        # Initialisierung
        manager = EnhancedVectorStoreManager(base_path)
        logger.info("Vector Store Manager initialisiert")
        
        # Lese Text
        try:
            with open(text_path, 'r', encoding='utf-8') as f:
                text = f.read()
        except UnicodeDecodeError:
            with open(text_path, 'r', encoding='latin1') as f:
                text = f.read()
                
        logger.info(f"Text erfolgreich gelesen: {text_path}")
        
        # Verarbeite Text
        print("Verarbeite Text...")
        documents, metrics = manager.process_text(text)
        manager.print_metrics(metrics)
        
        # Erstelle und speichere Stores
        print("\nErstelle und speichere Vector Stores...")
        stores = manager.create_and_save_stores(documents)
        
        # Validiere Stores
        manager.validate_stores(stores)
        logger.info("Stores erfolgreich validiert")
        
        # Test-Suche
        manager.test_search(stores)
        
        # Cleanup
        manager.cleanup()
        logger.info("Verarbeitung erfolgreich abgeschlossen")
        
    except Exception as e:
        logger.error(f"Fehler in der Hauptfunktion: {str(e)}")
        raise
    
if __name__ == "__main__":
    main()

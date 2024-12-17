import unittest
import numpy as np
import os
import tempfile
from sklearn.cluster import DBSCAN
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

class EnhancedVectorStoreManager:
    def __init__(self, base_path: str):
        self.base_path = base_path
        self.text_splitter = self._initialize_text_splitter()
        self.embeddings = self._initialize_embeddings()
        self.cluster_params = {
            'min_cluster_size': 5,
            'min_samples': 3,
            'metric': 'cosine',
            'eps': 0.3
        }
        self._embedding_cache = {}
        self.nlp = spacy.load("de_core_news_sm")
        
        # Initialisiere Store-Attribute als None
        self.vectorstore = None
        self.bm25 = None
        self.documents = None
        self.cluster_index = None
        
    def _initialize_text_splitter(self) -> RecursiveCharacterTextSplitter:
        """Initialisiert einen verbesserten TextSplitter"""
        return RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=150,  # Erhöhter Overlap für besseren Kontext
            separators=["\n\n", "\n", ". ", "? ", "! ", ";", ":", " - ", ",", " "],
            length_function=len,
        )
        
    def _initialize_embeddings(self) -> HuggingFaceEmbeddings:
        """Initialisiert das Embedding-Modell mit GPU-Unterstützung"""
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {device}")
        return HuggingFaceEmbeddings(
            model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
            model_kwargs={'device': device}
        )

    @lru_cache(maxsize=1000)
    def _embed_text(self, text: str) -> np.ndarray:
        """Cached Embedding-Generierung"""
        return self.embeddings.embed_query(text)

    def _calculate_chunk_importance(self, chunk: str, all_chunks: List[str]) -> float:
        """Berechnet Wichtigkeits-Score für einen Chunk"""
        words = chunk.lower().split()
        unique_words = set(words)
        importance = 0.0
        
        for word in unique_words:
            # Term Frequency
            tf = words.count(word) / len(words)
            # Document Frequency
            df = sum(1 for doc in all_chunks if word in doc.lower())
            # TF-IDF
            importance += tf * np.log(len(all_chunks) / (df + 1))
            
        return float(importance)

    def process_text(self, text: str) -> Tuple[List[Document], Dict[str, Any]]:
        """Verbesserte Textverarbeitung mit zusätzlichen Metriken"""
        try:
            print("Starte Textverarbeitung...")
            chunks = self.text_splitter.split_text(text)
            
            chunk_embeddings = []
            documents = []
            metrics = {
                'avg_chunk_size': 0,
                'total_chunks': len(chunks),
                'importance_scores': []
            }
            
            print(f"Verarbeite {len(chunks)} Chunks...")
            
            # Berechne Wichtigkeits-Scores
            importance_scores = [
                self._calculate_chunk_importance(chunk, chunks)
                for chunk in chunks
            ]
            
            for i, (chunk, importance) in enumerate(zip(chunks, importance_scores)):
                if i % 10 == 0:
                    print(f"Verarbeite Chunk {i+1} von {len(chunks)}")
                
                embedding = self._embed_text(chunk)
                chunk_embeddings.append(embedding)
                
                doc = Document(
                    page_content=chunk,
                    metadata={
                        "chunk_id": f"chunk_{i}",
                        "chunk_size": len(chunk),
                        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                        "embedding": embedding,
                        "importance_score": importance
                    }
                )
                documents.append(doc)
                metrics['avg_chunk_size'] += len(chunk)
                metrics['importance_scores'].append(importance)
            
            print("Führe Clustering durch...")
            cluster_labels = self._perform_clustering(np.array(chunk_embeddings))
            
            # Berechne Clustering-Qualität
            if len(set(cluster_labels)) > 1:
                silhouette = silhouette_score(np.array(chunk_embeddings), cluster_labels)
                metrics['silhouette_score'] = silhouette
                print(f"Clustering-Qualität (Silhouette Score): {silhouette:.3f}")
            
            for doc, label in zip(documents, cluster_labels):
                doc.metadata['cluster'] = int(label)
            
            metrics['avg_chunk_size'] /= len(chunks)
            metrics['num_clusters'] = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
            metrics['avg_importance_score'] = np.mean(metrics['importance_scores'])
            
            print(f"Clustering abgeschlossen. {metrics['num_clusters']} Cluster gefunden.")
            
            logging.info(f"Text erfolgreich verarbeitet: {metrics}")
            
            return documents, metrics
        
        except Exception as e:
            logging.error(f"Fehler bei der Textverarbeitung: {str(e)}")
            raise

    def _perform_clustering(self, embeddings: np.ndarray) -> np.ndarray:
        """Führt DBSCAN-Clustering durch"""
        return DBSCAN(
            eps=self.cluster_params['eps'],
            min_samples=self.cluster_params['min_samples'],
            metric=self.cluster_params['metric']
        ).fit(embeddings).labels_

    def create_and_save_stores(self, documents: List[Document]) -> Dict[str, Any]:
        """Erstellt und speichert erweiterte Vector Stores"""
        try:
            print("Erstelle Vector Stores...")
            
            # Lösche existierende Store-Dateien
            store_names = ['vectorstore', 'bm25', 'documents', 'cluster_index']
            for name in store_names:
                path = os.path.join(self.base_path, f'{name}.pkl')
                if os.path.exists(path):
                    os.remove(path)
                    print(f"Existierender Store {name} gelöscht.")
            
            # Speichere als Instanzattribute
            self.vectorstore = FAISS.from_documents(documents, self.embeddings)
            print("FAISS Vector Store erstellt.")
            
            tokenized_chunks = [doc.page_content.split() for doc in documents]
            self.bm25 = BM25Okapi(tokenized_chunks)
            print("BM25 Index erstellt.")
            
            self.documents = documents
            
            cluster_index = defaultdict(list)
            for i, doc in enumerate(documents):
                cluster_index[doc.metadata['cluster']].append(i)
            self.cluster_index = dict(cluster_index)
            
            stores = {
                'vectorstore': self.vectorstore,
                'bm25': self.bm25,
                'documents': self.documents,
                'cluster_index': self.cluster_index
            }
            
            print("Speichere neue Stores...")
            for name, store in stores.items():
                path = os.path.join(self.base_path, f'{name}.pkl')
                with open(path, 'wb') as f:
                    pickle.dump(store, f, protocol=pickle.HIGHEST_PROTOCOL)
                logging.info(f"{name} gespeichert: {path}")
                print(f"{name} gespeichert.")
            
            return stores
            
        except Exception as e:
            logging.error(f"Fehler beim Speichern: {str(e)}")
            raise

    def hybrid_search(self, query: str, k: int = 5) -> List[Dict]:
        """Implementiert gewichtete Hybrid-Suche"""
        try:
            # Validiere Stores
            self.validate_stores()
            # Vector-Suche
            vector_results = self.vectorstore.similarity_search_with_score(query, k=k)
            
            # BM25-Suche
            bm25_scores = self.bm25.get_scores(query.split())
            
            # Gewichte für verschiedene Faktoren
            weights = {
                'vector': 0.6,
                'bm25': 0.3,
                'cluster': 0.1,
                'importance': 0.2
            }
            
            combined_results = []
            for doc, vector_score in vector_results:
                idx = int(doc.metadata['chunk_id'].split('_')[1])
                bm25_score = bm25_scores[idx]
                
                # Normalisiere Scores
                vector_score_norm = 1 - (vector_score / max(1e-6, vector_score))
                bm25_score_norm = bm25_score / max(1e-6, max(bm25_scores))
                
                # Cluster-Bonus
                cluster_bonus = weights['cluster'] if doc.metadata['cluster'] != -1 else 0
                
                # Wichtigkeits-Bonus
                importance_bonus = doc.metadata.get('importance_score', 0) * weights['importance']
                
                # Kombinierter Score
                combined_score = (
                    weights['vector'] * vector_score_norm +
                    weights['bm25'] * bm25_score_norm +
                    cluster_bonus +
                    importance_bonus
                )
                
                combined_results.append({
                    'content': doc.page_content,
                    'score': combined_score,
                    'metadata': doc.metadata
                })
            
            return sorted(combined_results, key=lambda x: x['score'], reverse=True)
            
        except Exception as e:
            logging.error(f"Fehler bei der Suche: {str(e)}")
            raise
    def validate_stores(self):
        """Validiert dass alle notwendigen Stores geladen sind"""
        required_attrs = ['vectorstore', 'bm25', 'documents', 'cluster_index']
        missing_attrs = [
            attr for attr in required_attrs 
            if not hasattr(self, attr) or getattr(self, attr) is None
        ]
        
        if missing_attrs:
            raise ValueError(
                f"Folgende Stores fehlen oder sind None: {', '.join(missing_attrs)}"
            )

def main():
    """Hauptfunktion"""
    try:
        print("Starte erweiterte Verarbeitung...")
        start_time = time.time()
    
        manager = EnhancedVectorStoreManager(SAVE_PATH)
        
        # Versuche Text zu lesen
        encodings = ['utf-8', 'latin1', 'cp1252']
        text = None
        
        for encoding in encodings:
            try:
                with open(TEXT_PATH, 'r', encoding=encoding) as f:
                    text = f.read()
                print(f"Datei gelesen mit Encoding: {encoding}")
                break
            except UnicodeDecodeError:
                continue
        
        if text is None:
            raise Exception("Konnte die Datei nicht lesen")
        
        # Verarbeite Text und erstelle Stores
        print("Erstelle neue Stores...")
        documents, metrics = manager.process_text(text)
        stores = manager.create_and_save_stores(documents)
        
        print("\nVerarbeitungsmetriken:")
        print(f"Chunks: {metrics['total_chunks']}")
        print(f"Durchschnittliche Chunk-Größe: {metrics['avg_chunk_size']:.2f} Zeichen")
        print(f"Anzahl Cluster: {metrics['num_clusters']}")
        if 'silhouette_score' in metrics:
            print(f"Clustering-Qualität: {metrics['silhouette_score']:.3f}")
        print(f"Durchschnittliche Wichtigkeit: {metrics['avg_importance_score']:.3f}")
        
        # Test-Suche
        test_query = "Wirtschaftsprüfung"
        print(f"\nFühre Test-Suche durch: '{test_query}'")
        results = manager.hybrid_search(test_query, k=3)
        
        print("\nTop 3 Ergebnisse:")
        for i, result in enumerate(results, 1):
            print(f"\n{i}. Score: {result['score']:.3f}")
            print(f"Text: {result['content'][:200]}...")
        
        end_time = time.time()
        print(f"\nVerarbeitung abgeschlossen in {end_time - start_time:.2f} Sekunden")
            
    except Exception as e:
        logging.error(f"Fehler in main: {str(e)}")
        print(f"Fehler: {str(e)}")
        raise
if __name__ == "__main__":
    main()
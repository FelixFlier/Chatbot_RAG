# optimized_store.py
import pickle
from pathlib import Path
from typing import Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)

class OptimizedVectorStore:
    def __init__(self, file_path: Path):
        self.file_path = file_path
        self._store = None
        self._docstore = None
        self._index_to_docstore_id = None
        
    @property
    def store(self):
        """Lazy Loading des Vector Stores"""
        if self._store is None:
            self._load_store()
        return self._store
    
    @property
    def docstore(self):
        """Lazy Loading des Docstores"""
        if self._store is None:
            self._load_store()
        return self._docstore
    
    @property
    def index_to_docstore_id(self):
        """Lazy Loading der Index-Mapping"""
        if self._store is None:
            self._load_store()
        return self._index_to_docstore_id
    
    def _load_store(self):
        """Lädt den Vector Store bei Bedarf"""
        if not self.file_path.exists():
            raise FileNotFoundError(f"Vector Store nicht gefunden: {self.file_path}")
            
        logger.info(f"Lade Vector Store von {self.file_path}")
        try:
            with open(self.file_path, 'rb') as f:
                self._store = pickle.load(f)
                # Extrahiere wichtige Komponenten
                if hasattr(self._store, 'docstore'):
                    self._docstore = self._store.docstore
                if hasattr(self._store, 'index_to_docstore_id'):
                    self._index_to_docstore_id = self._store.index_to_docstore_id
        except Exception as e:
            logger.error(f"Fehler beim Laden des Vector Stores: {str(e)}")
            raise
            
    def similarity_search_with_relevance_scores(self, *args, **kwargs):
        """Delegiert Suche an den Store"""
        return self.store.similarity_search_with_relevance_scores(*args, **kwargs)
    
    def similarity_search(self, *args, **kwargs):
        """Delegiert Suche an den Store"""
        return self.store.similarity_search(*args, **kwargs)

    @property
    def embedding_function(self):
        """Delegiert Embedding-Funktion"""
        return self.store.embedding_function

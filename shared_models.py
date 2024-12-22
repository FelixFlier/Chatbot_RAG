# shared_models.py
import spacy
import logging
from typing import Optional
import numpy as np
from langchain_huggingface import HuggingFaceEmbeddings

logger = logging.getLogger(__name__)

class SharedModels:
    _instance = None
    _spacy_model = None
    _embeddings_model = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(SharedModels, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if not self._initialized:
            self._initialized = True
            self._spacy_model = None
            self._embeddings_model = None
            logger.info("SharedModels initialisiert")
    
    @property
    def spacy_model(self):
        """Singleton-Zugriff auf das Spacy-Modell"""
        if self._spacy_model is None:
            logger.info("Initialisiere Spacy-Modell...")
            try:
                self._spacy_model = spacy.load("de_core_news_sm")
                logger.info("Spacy-Modell erfolgreich geladen")
            except Exception as e:
                logger.error(f"Fehler beim Laden des Spacy-Modells: {str(e)}")
                raise
        return self._spacy_model
    
    @property
    def embeddings_model(self):
        """Singleton-Zugriff auf das Embeddings-Modell"""
        if self._embeddings_model is None:
            logger.info("Initialisiere Embeddings-Modell...")
            try:
                self._embeddings_model = HuggingFaceEmbeddings(
                    model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
                )
                logger.info("Embeddings-Modell erfolgreich geladen")
            except Exception as e:
                logger.error(f"Fehler beim Laden des Embeddings-Modells: {str(e)}")
                raise
        return self._embeddings_model

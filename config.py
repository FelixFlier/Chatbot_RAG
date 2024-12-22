# config.py
from pathlib import Path
import streamlit as st
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import GoogleGenerativeAI
import pickle

# Konstanten
BASE_PATH = Path(__file__).resolve().parent
VECTOR_STORE_PATH = BASE_PATH / "vector_stores"
VECTOR_STORE_PATH.mkdir(parents=True, exist_ok=True)

# Cache-Funktionen
@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
    )

@st.cache_resource
def load_llm():
    return GoogleGenerativeAI(
        model="gemini-pro",
        temperature=0.4,
        top_p=0.95,
        max_output_tokens=700
    )

@st.cache_data
def load_vector_stores():
    """Lädt Vector Store und BM25 mit Caching"""
    with open(VECTOR_STORE_PATH / "vectorstore.pkl", 'rb') as f:
        vectorstore = pickle.load(f)
    with open(VECTOR_STORE_PATH / "bm25.pkl", 'rb') as f:
        bm25 = pickle.load(f)
    return vectorstore, bm25

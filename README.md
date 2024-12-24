# RAG-Chatbot: Intelligenter Assistent für Wirtschaftsprüfungsthemen

## 🎯 Projektübersicht
Der WP-RAG-Chatbot ist ein hochmoderner, KI-gestützter Assistent, spezialisiert auf Wirtschaftsprüfungsthemen. Basierend auf der RAG-Architektur (Retrieval-Augmented Generation) kombiniert er fortschrittliche Technologien wie Google's Gemini Pro LLM mit optimierten Embedding-Modellen für präzise, kontextbezogene Antworten.

### 🌟 Kernfunktionen
- **Intelligente Themenerkennung & -verfolgung**
  - Dynamische Erkennung von Themenwechseln
  - Kontinuierliche Kontextaktualisierung
  - Gewichtete Themenrelevanz
  - Hierarchische Themenmodellierung

- **Innovative Sucharchitektur**
  - Mehrstufiger hybrider Suchalgorithmus
  - Semantische und graphbasierte Suche
  - Statistische und keyword-basierte Analyse
  - Dynamische Gewichtungsanpassung
  - Kontextbewusstes Ranking

- **Fortschrittliche Qualitätssicherung**
  - Automatische Qualitätsmetriken
  - Kontinuierliche Antwortoptimierung
  - Kontextuelle Relevanzprüfung
  - Datenkonsistenzprüfung

- **Benutzerfreundliches Interface**
  - Adaptives Light/Dark Design
  - Intuitive Bedienelemente
  - Responsive Layoutanpassung
  - Echtzeit-Feedback-System

## 🛠 Technische Details

### Verwendete Technologien
- **Frontend-Framework**: 
  - Streamlit (moderne Python Web-App)
  - Responsive CSS/HTML
  - Custom Theme Engine
  - Dynamisches State Management

- **KI & Machine Learning**:
  - Google Gemini Pro LLM
  - Sentence-Transformers (multilingual-mpnet-base-v2)
  - SpaCy NLP Pipeline
  - Custom Neural Networks

- **Datenverarbeitung**:
  - FAISS Vectorstore
  - BM25 Textindexierung
  - NetworkX Graph Processing
  - Optimierte Sparse Matrices

### 🔍 Detaillierte Suchalgorithmus-Architektur

#### Mehrstufige Hybridsuche (Enhanced Hybrid Search)

1. **Stage 1: Schnelle Vorauswahl**
   - **Semantische Suche**:
     - Kosinus-Ähnlichkeitsberechnung mit FAISS
     - Embedding-Cache für häufige Anfragen
     - Normalisierte Vektorrepräsentationen
   
   - **Keyword-basierte Suche**:
     - Optimierte BM25-Implementierung
     - TF-IDF Gewichtung
     - N-Gram Analyse

2. **Stage 2: Erweiterte Analyse** (bei Bedarf)
   - **Graph-basierte Suche**:
     - PageRank-Algorithmus für Dokumentrelevanz
     - Thematische Cluster-Analyse
     - Gewichtete Kantenbewertung
   
   - **Statistische Suche**:
     - Sparse Matrix Operationen
     - Adaptive Schwellenwerte
     - Hierarchische Dokumentbeziehungen

3. **Stage 3: Ergebnisverfeinerung**
   - Dynamische Gewichtungsanpassung
   - Kontextuelle Relevanzprüfung
   - Diversity-basiertes Reranking

### 🧠 Intelligente Kontextverarbeitung

#### Thematisches Tracking
1. **Dynamische Themenerkennung**
   - Embedding-basierte Themenextraktion
   - Zeitbasierte Gewichtung
   - Hierarchische Themenmodellierung

2. **Kontextmanagement**
   - Sliding Window für aktive Themen
   - Gewichtetes Themen-Decay
   - Cross-Reference Analyse

#### Intent Recognition System
- **Multi-Layer Intent Analyse**:
  - Pattern-basierte Vorerkennung
  - LLM-basierte Feinanalyse
  - Konfidenzscoring

### 💾 Datenverwaltung

#### Vector Store Management
- Optimierte FAISS-Integration
- Inkrementelle Updates
- Automatische Index-Optimierung

#### Dokumentenverarbeitung
- Hierarchische Dokumentstruktur
- Metadaten-Management
- Versionscontrolling

### 🔧 Systemarchitektur

#### Core Components
1. **IntelligentRAGChatbot**
   - Konversationsmanagement
   - System-Orchestrierung
   - Error-Handling

2. **EnhancedHybridSearcher**
   - Such-Koordination
   - Performance-Monitoring
   - Adaptive Optimierung

3. **ResponseGenerator**
   - Template-Management
   - Qualitätssicherung
   - Formatierung

### 📊 Qualitätsmetriken

#### Antwortqualität
- Datenkonsistenz
- Query-Relevanz
- Kontext-Abdeckung
- Thematische Kohärenz

## 🚀 Setup & Installation

### Systemanforderungen
- Python 3.8+
- 8GB RAM (minimum)
- CUDA-fähige GPU (optional)

### Hauptabhängigkeiten
- streamlit
- langchain
- sentence-transformers
- spacy
- faiss-cpu/faiss-gpu
- networkx

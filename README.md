# RAG-Chatbot: Intelligenter Assistent für Wirtschaftsprüfungsthemen

## 🎯 Projektübersicht
Der WP-RAG-Chatbot ist ein hochmoderner, KI-gestützter Assistent, spezialisiert auf Wirtschaftsprüfungsthemen. Basierend auf der RAG-Architektur (Retrieval-Augmented Generation) kombiniert er fortschrittliche Technologien wie Google's Gemini Pro LLM mit optimierten Embedding-Modellen für präzise, kontextbezogene Antworten.

### 🌟 Kernfunktionen
- **Intelligente Themenerkennung & -verfolgung**
  - Dynamische Erkennung von Themenwechseln
  - Kontinuierliche Kontextaktualisierung
  - Gewichtete Themenrelevanz

- **Innovative Sucharchitektur**
  - Mehrstufiger hybrider Suchalgorithmus
  - Semantische und graphbasierte Suche
  - Statistische und keyword-basierte Analyse
  - Dynamische Gewichtungsanpassung

- **Fortschrittliche Qualitätssicherung**
  - Automatische Qualitätsmetriken
  - Kontinuierliche Antwortoptimierung
  - Kontextuelle Relevanzprüfung

- **Benutzerfreundliches Interface**
  - Adaptives Light/Dark Design
  - Intuitive Bedienelemente
  - Responsive Layoutanpassung

## 🛠 Technische Details

### Verwendete Technologien
- **Frontend-Framework**: 
  - Streamlit (moderne Python Web-App)
  - Responsive CSS/HTML
  - Custom Theme Engine

- **KI & Machine Learning**:
  - Google Gemini Pro LLM
  - Sentence-Transformers (multilingual-mpnet-base-v2)
  - Custom Neural Networks

- **Datenverarbeitung**:
  - FAISS Vectorstore
  - SpaCy NLP Pipeline
  - NetworkX Graph Processing

### Systemarchitektur

#### IntelligentRAGChatbot (Kernmodul)
- **Funktionen**:
  - Konversationsmanagement
  - Themenverfolgung
  - Hybride Suchkoordination
  
- **Features**:
  - Cache-Optimierung
  - Performance-Monitoring
  - Fehlerbehandlung

#### EnhancedHybridSearcher
- **Suchstrategien**:
  - Semantische Ähnlichkeitssuche
  - Graphbasierte Traversierung
  - BM25 Keyword-Matching
  - Statistische Analyse

- **Optimierungen**:
  - Multi-Stage Caching
  - Adaptive Gewichtung
  - Parallele Verarbeitung

#### ResponseGenerator
- **Antwortgenerierung**:
  - Kontextuelle Templating
  - Dynamische Formatierung
  - Qualitätssicherung

- **Funktionen**:
  - Intent-basierte Anpassung
  - Automatische Korrektur
  - Style-Konsistenz

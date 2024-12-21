# RAG-Chatbot: Intelligenter Assistent für Wirtschaftsprüfungsthemen

## 🎯 Projektübersicht

WP-RAG-Chatbot ist ein fortschrittlicher, KI-gestützter Chatbot, der speziell für die Beantwortung von Fragen im Bereich Wirtschaftsprüfung entwickelt wurde. Der Chatbot basiert auf der RAG-Architektur (Retrieval-Augmented Generation) und nutzt modernste Technologien wie Google's Gemini Pro LLM und verschiedene Embedding-Modelle für präzise und kontextuell relevante Antworten.

### 🌟 Hauptfunktionen

- **Intelligente Themenerkennung**: Dynamische Erkennung und Verfolgung von Gesprächsthemen
- **Hybride Sucharchitektur**: Verwendet einen zweistufigen Suchalgorithmus der dabei semantische, graphen-basierte,Statistisch-basierte und keyword-basierte Suche kombiniert 
- **Qualitätssicherung**: Automatische Überprüfung der Antwortqualität
- **Adaptives Dialogmanagement**: Kontextbewusstes Gesprächsmanagement
- **Benutzerfreundliche Oberfläche**: Moderne Web-Interface mit Streamlit
- **Themenwechselerkennung**: Intelligente Erkennung von Kontextwechseln
- **Mehrsprachige Unterstützung**: Optimiert für deutsche Fachsprache

## 🛠 Technologie-Stack

- **Frontend**: Streamlit
- **LLM**: Google Gemini Pro
- **Embeddings**: Sentence-Transformers (paraphrase-multilingual-mpnet-base-v2)
- **Vektor-Datenbank**: FAISS
- **NLP-Verarbeitung**: SpaCy
- **Weitere Bibliotheken**: 
  - LangChain
  - NumPy
  - NetworkX
  - scikit-learn
  - RANK-BM25

## 📋 Systemanforderungen

- Python 3.8+
- 4GB RAM (minimal)
- Google API Key für Gemini Pro
- Internetverbindung für API-Zugriffe

## 🔋 Hauptkomponenten

### IntelligentRAGChatbot
- Kernklasse für die Chatbot-Funktionalität
- Verwaltet Konversationskontext und Themenverfolgung
- Implementiert hybride Suchstrategien

### DialogStateManager
- Steuert den Gesprächsfluss
- Verwaltet verschiedene Dialogphasen
- Passt Antwortstrategie dynamisch an

### EnhancedHybridSearcher
- Kombiniert verschiedene Suchmethoden
- Implementiert Cache-Strategien
- Optimiert Suchergebnisse durch Gewichtung

### ResponseGenerator
- Generiert kontextbezogene Antworten
- Prüft Antwortqualität
- Implementiert Reparaturstrategien

## 🔍 Funktionsweise

1. **Anfrageverarbeitung**:
   - Erkennung der Benutzerabsicht
   - Analyse der Konzepte
   - Aktualisierung des Themenkontexts

2. **Informationsabruf**:
   - Hybride Suche in der Wissensbasis
   - Kontextbasierte Filterung
   - Dynamische Gewichtung der Ergebnisse

3. **Antwortgenerierung**:
   - Kontextbezogene Formulierung
   - Qualitätsprüfung
   - Automatische Verbesserung bei Bedarf

## 🎨 UI-Funktionen

- Helles/Dunkles Design
- Anpassbare Schriftgröße
- Kopier-Funktion für Antworten
- Feedback-System
- Qualitätsmetriken-Anzeige
- Automatisches Scrollen

## 📊 Qualitätsmetriken

Der Chatbot überwacht kontinuierlich:
- Antwortrelevanz
- Vollständigkeit
- Kohärenz
- Genauigkeit
- Kontextabdeckung
# Arabic RAG Pipeline

A practical **Arabic Retrieval-Augmented Generation (RAG) pipeline** built with Python, Ollama, Qwen, Nomic Embeddings, and ChromaDB.

The project focuses on building a reliable Arabic question-answering system that retrieves relevant information from a knowledge base before generating an answer with an LLM.

## 🚀 Features

* Arabic text normalization and query correction
* Arabic query analysis
* Entity detection
* Intent detection
* Semantic search using Nomic Embeddings
* ChromaDB vector storage and retrieval
* Document re-ranking
* Keyword and semantic relevance scoring
* Topic and entity consistency checking
* Relation evidence detection
* Relevance Gate to filter weak results
* Context selection before generation
* Grounded answer generation using Qwen
* Local AI inference using Ollama

## 🧠 Pipeline

```text
User Question
      ↓
Arabic Query Processing
      ↓
Normalization & Correction
      ↓
Query Analysis
      ↓
Entity Detection
      ↓
Intent Detection
      ↓
Embedding Generation
      ↓
ChromaDB Retrieval
      ↓
Re-ranking
      ↓
Topic / Entity Consistency
      ↓
Relation Evidence
      ↓
Relevance Gate
      ↓
Context Selection
      ↓
Qwen
      ↓
Grounded Answer
```

## 🛠️ Tech Stack

* Python
* Ollama
* Qwen 2.5 Coder 7B
* Nomic Embed Text
* ChromaDB
* Semantic Search
* Retrieval-Augmented Generation (RAG)

## 📚 Knowledge Base

The current knowledge base is built from Python learning content from **Hsoub Academy** and is stored as vectorized chunks inside ChromaDB.

## 🎯 Project Goal

The goal of this project is to build a practical Arabic RAG system that does more than simple semantic retrieval by combining:

* Semantic similarity
* Keyword matching
* Entity matching
* Intent analysis
* Topic consistency
* Relation evidence
* Relevance filtering
* Context-aware generation

The project is being developed incrementally with a focus on improving retrieval accuracy and reducing unsupported or hallucinated answers.

## 📌 Current Status

The core RAG pipeline is working locally.

The current development focus is improving **grounded generation**, ensuring that the LLM generates answers strictly based on the retrieved context and does not introduce unsupported information.

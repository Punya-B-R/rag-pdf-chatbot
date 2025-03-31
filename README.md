# RAG PDF Chatbot

This project is a RAG (Retrieval-Augmented Generation) system that transforms PDFs into queryable knowledge sources. By combining Google's latest Gemini embeddings with efficient vector search, it delivers accurate, context-grounded answers from your documents.

## 1. Document Processing Pipeline

- **PDF Text Extraction:** Uses pdfplumber for layout-aware text parsing.
- **Semantic Chunking:** Employs Langchain's RecursiveCharacterTextSplitter to split text into 1000-character chunks with a 200-character overlap.
- **HyDE Optimization:** Generates hypothetical answers to improve retrieval accuracy.

## 2. Vector Search Engine

- **Gemini Embeddings:** Creates 768-dimension vectors using the `models/embedding-001`.
- **ChromaDB Indexing:** Provides persistent storage with cosine similarity search.
- **Contextual Retrieval:** Retrieves the top-3 most relevant text chunks per query.

## 3. Generative Q&A

- **Gemini-1.5-Flash:** Generates answers that are constrained to the retrieved context.
- **Two-Phase Prompting:**
  - *Retrieval-focused Query Rewriting:* Enhances query relevance.
  - *Context-aware Response Generation:* Produces detailed responses with citation prompts.

## 4. Streamlit Interface

- **Session State:** Maintains chat history and caches processed documents.
- **Streaming Responses:** Simulated token-by-token output for a dynamic user experience.
- **Auto-Cleanup:** Manages temporary files using `atexit` for automatic cleanup.



1. **Clone the repository:**

   ```bash
   git clone https://github.com/your_username/rag-pdf-chatbot.git
   cd rag-pdf-chatbot

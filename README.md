# RAG PDF Chatbot

## Overview

This project implements a PDF chatbot that allows users to interact with the content of their PDF documents using a conversational interface.

## Key Features

- **PDF Upload & Processing**  
  Upload PDFs through a simple Streamlit interface. The uploaded PDF is processed to extract text, which is then split into semantically meaningful chunks.

- **Semantic Text Chunking**  
  Utilizes Langchain's `RecursiveCharacterTextSplitter` to divide text into manageable pieces while preserving semantic context.

- **Embedding Generation & Storage**  
  Uses Google Generative AI to generate embeddings for each text chunk. These embeddings are stored in ChromaDB for efficient retrieval.

- **Conversational Query Interface**  
  A chat-based UI built with Streamlit enables users to ask questions about the PDF content. The application retrieves relevant information and generates context-aware responses.

- **Retrieval Augmented Generation (RAG)**  
  The system retrieves the most contextually relevant text chunks from the document and then leverages generative AI to produce precise, context-aware responses. HyDE Optimization further refines this process by generating hypothetical answers that guide the retrieval phase, ensuring that the best possible context is used for the final output.

## Dependencies

- **Streamlit** – For the web-based chat interface.
- **PDFPlumber** – To extract text from PDF documents.
- **ChromaDB** – For embedding storage and fast retrieval.
- **Google Generative AI (google-generativeai)** – To generate embeddings and conversational responses.
- **Python-Dotenv** – To load environment variables.
- **Langchain** – For semantic text splitting.

## Local Setup Guide
### 1. **Install Python 3.9+**  
   Download from [python.org](https://www.python.org/downloads/) then verify:
   ```bash
   python --version
   ```

### 2. Clone the Repository
   ```bash
   git clone <repository-url>
   cd <repository-folder-name>
   ```

### 3. Set Up Virtual Environment
   ```bash
   python -m venv venv
   venv\Scripts\activate # Windows
   source venv/bin/activate # macOS/Linux
   ```

### 4. Install Dependencies
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```
### 5. Configure API Key
   - In project root, create a new file named `.env`
   - Get key from [Google AI Studio](https://aistudio.google.com/app/apikey)  
   - Add to `.env`:
     ```text
     GEMINI_API_KEY="YOUR_API_KEY"  # Paste your key here
     ```
     
### 6. Run the Application
   ```bash
   streamlit run app.py
   ```
   The application will start running at http://localhost:8501.


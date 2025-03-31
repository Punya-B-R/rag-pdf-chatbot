import os
import time
import atexit
import shutil
from pathlib import Path

import streamlit as st
import chromadb
import pdfplumber
import google.generativeai as genai
from dotenv import load_dotenv
from google.generativeai import configure
from chromadb.utils import embedding_functions

from utils.file_utils import save_uploaded_file
from utils.text_utils import chunk_text

# CONFIGURATION & SETUP

# Streamlit page
st.set_page_config(
    page_title="RAG PDF Chatbot", 
    page_icon="📑", 
    initial_sidebar_state="collapsed"
)

# Load environment variables
load_dotenv()

# Configure Gemini API
configure(api_key=os.getenv("GEMINI_API_KEY"))

# UTILITY FUNCTIONS

def cleanup_uploads():
    """
    Cleanup function to delete contents of uploads/ directory on exit.
    Preserves the uploads directory itself, only removes its contents.
    """
    uploads_dir = Path("uploads")
    try:
        for item in uploads_dir.glob("*"):
            if item.is_file():
                item.unlink()  # Delete file
            elif item.is_dir():
                shutil.rmtree(item)  # Delete directory
        print("✓ Cleared uploads contents")
    except Exception as e:
        print(f"⚠️ Cleanup warning: {str(e)}")

# Register cleanup function to run at exit
atexit.register(cleanup_uploads)

# MAIN APPLICATION

def initialize_chroma_client():
    """
    Initialize and clean up ChromaDB client.
    Deletes any existing collections to start fresh.
    """
    chroma_client = chromadb.PersistentClient(path="chroma_db")
    
    try:
        collections = chroma_client.list_collections()
        for col in collections:
            try:
                chroma_client.delete_collection(name=col.name)
            except Exception as e:
                if "only returns collection names" not in str(e):
                    st.warning(f"Couldn't delete {col.name}: {str(e)}")
    except Exception as e:
        st.warning(f"Cleanup skipped: {str(e)}")
    
    return chroma_client

def process_pdf_file(uploaded_file, chroma_client):
    """
    Process the uploaded PDF file:
    1. Extract text
    2. Chunk text
    3. Generate embeddings
    4. Store in ChromaDB
    """
    with st.spinner("Processing PDF..."):
        # Save and read PDF
        pdf_path = save_uploaded_file(uploaded_file)
        text = ""
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                text += page.extract_text() + "\n"
        
        # Chunk text
        chunks = chunk_text(text)
        st.info(f"✓ Split into {len(chunks)} chunks")
        
        # Generate embeddings
        embeddings = []
        for chunk in chunks:
            response = genai.embed_content(
                model="models/embedding-001",
                content=chunk,
                task_type="retrieval_document"
            )
            embeddings.append(response['embedding'])
        
        # Create ChromaDB collection
        collection_name = f"doc_{abs(hash(uploaded_file.name))}"
        collection = chroma_client.get_or_create_collection(
            name=collection_name,
            embedding_function=embedding_functions.GoogleGenerativeAiEmbeddingFunction(
                api_key=os.getenv("GEMINI_API_KEY"),
                model_name="models/embedding-001"
            )
        )
        
        # Add documents to collection
        metadatas = [{"chunk_num": i} for i in range(len(chunks))]
        collection.add(
            ids=[str(i) for i in range(len(chunks))],  
            embeddings=embeddings,
            documents=chunks,
            metadatas=metadatas  
        )
        
        # Store in session state
        st.session_state.processed_data = {
            "chunks": chunks,
            "collection": collection,
            "pdf_name": uploaded_file.name
        }
        st.success("PDF ready!")

def generate_response(prompt, collection):
    """
    Generate response using HyDE (Hypothetical Document Embeddings) technique:
    1. Generate hypothetical answer
    2. Use it to retrieve relevant chunks
    3. Generate final response with context
    """
    # Initialize the GenerativeModel
    model = genai.GenerativeModel('gemini-2.0-flash')
    
    # Generate Hypothetical Answer (HyDE)
    hyde_prompt = f"""Generate a comprehensive hypothetical answer that might exist in the document for:
    Question: {prompt}
    Include key terms and concepts the document would contain:"""
    
    hyde_response = model.generate_content(hyde_prompt)
    hyde_answer = hyde_response.text if hasattr(hyde_response, 'text') else prompt
    
    # Retrieve Relevant Chunks
    results = collection.query(
        query_texts=[hyde_answer],
        n_results=3,
        include=["documents", "distances"]
    )
    
    # Generate Context-Aware Response
    context = "\n\n---\n\n".join(results["documents"][0])
    
    enhanced_prompt = f"""Answer this question based ONLY on the following context:
    Question: {prompt}
    Context: {context}
    
    Instructions:
    - Provide a concise and direct answer first.
    - Follow the direct answer with a detailed explanation using relevant information from the context.
    - Mention supporting facts or key concepts to justify the response.
    - Say "I don't know" if the context does not provide enough information.
    - Never hallucinate information.

    Format:
    - Key Insight: [Concise Answer]
    - Additional Insights: [Detailed Explanation]
    """
    
    response = model.generate_content(enhanced_prompt)
    return response.text if hasattr(response, 'text') else "Sorry, I couldn't generate a response."

def display_chat_interface(collection):
    """
    Display chat interface and handle conversation flow
    """
    st.divider()
    st.subheader("Chat with your PDF")
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Handle user input
    if prompt := st.chat_input("Ask about the PDF"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        try:
            with st.spinner("Analyzing..."):
                # Generate response
                full_response = generate_response(prompt, collection)
                
                # Display assistant response with streaming effect
                with st.chat_message("assistant"):
                    placeholder = st.empty()
                    streamed_text = ""
                    for token in full_response.split(" "):
                        streamed_text += token + " "
                        placeholder.markdown(streamed_text)
                        time.sleep(0.1)  # Simulate streaming

                # Add assistant response to chat history
                st.session_state.messages.append({"role": "assistant", "content": full_response})

        except Exception as e:
            error_msg = f"Error processing query: {str(e)}"
            st.error(error_msg)
            st.session_state.messages.append({"role": "assistant", "content": error_msg})

def main():
    """
    Main application function
    """
    # Initialize ChromaDB client
    chroma_client = initialize_chroma_client()
    
    # Set up UI
    st.title("📚 RAG PDF Chatbot")
    uploaded_file = st.file_uploader("Upload PDF", type="pdf")

    if uploaded_file:
        try:
            # Process PDF if not already processed
            if ("processed_data" not in st.session_state or 
                st.session_state.processed_data["pdf_name"] != uploaded_file.name):
                process_pdf_file(uploaded_file, chroma_client)
            else:
                st.success("✓ Using previously processed PDF")

            # Display chat interface
            display_chat_interface(st.session_state.processed_data["collection"])

        except Exception as e:
            st.error(f"Error: {str(e)}")
            st.stop()

if __name__ == "__main__":
    main()
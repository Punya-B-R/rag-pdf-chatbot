import os
import streamlit as st
import chromadb
import pdfplumber
import google.generativeai as genai
from dotenv import load_dotenv
from google.generativeai import configure
from utils.file_utils import save_uploaded_file
from utils.text_utils import chunk_text
from chromadb.utils import embedding_functions
import shutil
from pathlib import Path
import atexit

def cleanup_uploads():
    """Delete only the contents of uploads/ without removing the folder"""
    uploads_dir = Path("uploads")
    try:
        # Delete all files and subdirectories inside uploads/
        for item in uploads_dir.glob("*"):
            if item.is_file():
                item.unlink()  # Delete file
            elif item.is_dir():
                shutil.rmtree(item)  # Delete subdirectory
        print("✓ Cleared uploads contents (folder kept)")
    except Exception as e:
        print(f"⚠️ Cleanup warning: {str(e)}")

# Register cleanup
atexit.register(cleanup_uploads)
# Load environment variables
load_dotenv()

# Initialize Gemini
configure(api_key=os.getenv("GEMINI_API_KEY"))

def main():
    # Initialize ChromaDB client
    chroma_client = chromadb.PersistentClient(path="chroma_db")
    
    # Cleanup previous collections (silent ignore for Chroma v0.6.0+ warning)
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

    # Streamlit UI
    st.title("PDF Chatbot with Gemini")
    uploaded_file = st.file_uploader("Upload PDF", type="pdf")

    if uploaded_file:
        try:
            if "processed_data" not in st.session_state or st.session_state.processed_data["pdf_name"] != uploaded_file.name:
                with st.status("Processing PDF...", expanded=True) as status:
                    # Save and extract text
                    pdf_path = save_uploaded_file(uploaded_file)
                    text = ""
                    with pdfplumber.open(pdf_path) as pdf:
                        for page in pdf.pages:
                            text += page.extract_text() + "\n"
                    
                    # Chunk text
                    chunks = chunk_text(text)
                    status.write(f"✓ Split into {len(chunks)} chunks")
                    
                    # Generate embeddings
                    embeddings = []
                    for chunk in chunks:
                        response = genai.embed_content(
                            model="models/embedding-001",
                            content=chunk,
                            task_type="retrieval_document"
                        )
                        embeddings.append(response['embedding'])
                    
                    # Store in ChromaDB
                    collection_name = f"doc_{abs(hash(uploaded_file.name))}"
                    collection = chroma_client.get_or_create_collection(
                        name=collection_name,
                        embedding_function=embedding_functions.GoogleGenerativeAiEmbeddingFunction(
                            api_key=os.getenv("GEMINI_API_KEY"),
                            model_name="models/embedding-001"
                        )
                    )
                    metadatas = [{"chunk_num": i} for i in range(len(chunks))]

                    collection.add(
                        ids=[str(i) for i in range(len(chunks))],  # String IDs required
                        embeddings=embeddings,
                        documents=chunks,
                        metadatas=metadatas  # Optional but must match length if included
                    )
                    
                    # Cache processed data
                    st.session_state.processed_data = {
                        "chunks": chunks,
                        "collection": collection,
                        "pdf_name": uploaded_file.name
                    }
                    status.update(label="PDF ready!", state="complete", expanded=False)
            else:
                st.success("✓ Using previously processed PDF")

            # Chat interface
            st.divider()
            st.subheader("Chat with your PDF")
            
            if "messages" not in st.session_state:
                st.session_state.messages = []

            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])

            if prompt := st.chat_input("Ask about the PDF"):
                st.session_state.messages.append({"role": "user", "content": prompt})
                with st.chat_message("user"):
                    st.markdown(prompt)
    
                try:
                    with st.spinner("Analyzing..."):
                        # 1. Initialize the GenerativeModel
                        model = genai.GenerativeModel('gemini-2.0-flash')
                        
                        # 2. Generate Hypothetical Answer (HyDE)
                        hyde_prompt = f"""Generate a comprehensive hypothetical answer that might exist in the document for:
                        Question: {prompt}
                        Include key terms and concepts the document would contain:"""
                        
                        hyde_response = model.generate_content(hyde_prompt)
                        hyde_answer = hyde_response.text if hasattr(hyde_response, 'text') else prompt  # Fallback if no HyDE response
                        
                        # 3. Retrieve Relevant Chunks
                        data = st.session_state.processed_data
                        results = data["collection"].query(
                            query_texts=[hyde_answer],
                            n_results=3,
                            include=["documents", "distances"]
                        )
                        
                        # 4. Generate Context-Aware Response
                        context = "\n\n---\n\n".join(results["documents"][0])
                        
                        # Updated Enhanced Prompt for Descriptive Answers
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
                        response_text = response.text if hasattr(response, 'text') else "Sorry, I couldn't generate a response."

                except Exception as e:
                    response_text = f"Error processing query: {str(e)}"
                
                # Display assistant response
                with st.chat_message("assistant"):
                    st.markdown(response_text)
                st.session_state.messages.append({"role": "assistant", "content": response_text})

        except Exception as e:
            st.error(f"Error: {str(e)}")
            st.stop()

if __name__ == "__main__":
    main()
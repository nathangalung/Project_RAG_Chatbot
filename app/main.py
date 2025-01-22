import sys
import os
import torch
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
from core.llm import get_llm, get_chat_response
from core.embeddings import get_embeddings
from processors.document_loader import load_document
from processors.prompt_templates import get_qa_template
from langchain_community.vectorstores import FAISS

def check_gpu():
    if torch.cuda.is_available():
        st.sidebar.success(f"Using GPU: {torch.cuda.get_device_name(0)}")
        torch.cuda.set_per_process_memory_fraction(0.6)
        torch.cuda.empty_cache()
    else:
        st.sidebar.warning("No GPU detected, running on CPU")

def process_document(uploaded_file, embeddings):
    try:
        docs = load_document(uploaded_file)
        chunks = []
        chunk_size = 5  # Process 5 documents at a time
        
        # Process first chunk to initialize vectorstore
        initial_chunk = docs[:chunk_size]
        vectorstore = FAISS.from_documents(
            initial_chunk, 
            embeddings,
            distance_strategy="cosine"
        )
        
        # Process remaining chunks
        for i in range(chunk_size, len(docs), chunk_size):
            chunk = docs[i:i + chunk_size]
            chunk_vectorstore = FAISS.from_documents(
                chunk, 
                embeddings,
                distance_strategy="cosine"
            )
            # Merge into main vectorstore
            vectorstore.merge_from(chunk_vectorstore)
            torch.cuda.empty_cache()
            
        return vectorstore
    except Exception as e:
        raise Exception(f"Error processing document: {str(e)}")

def main():
    st.title("Document Q&A Bot")
    
    check_gpu()
    
    @st.cache_resource
    def load_models():
        with torch.amp.autocast(device_type='cuda'):  # Updated autocast
            return get_llm(), get_embeddings()
    
    try:
        with st.spinner("Loading models..."):
            llm, embeddings = load_models()
            st.success("Models loaded successfully!")
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        return

    uploaded_file = st.file_uploader("Upload a PDF document", type="pdf")
    
    if uploaded_file:
        try:
            with st.spinner("Processing document..."):
                vectorstore = process_document(uploaded_file, embeddings)
                st.success("Document processed successfully!")
                
                user_question = st.text_input("Ask a question about your document:")
                
                if user_question:
                    with st.spinner("Thinking..."):
                        try:
                            # Get context with error handling
                            context_docs = vectorstore.similarity_search(
                                user_question, 
                                k=2,
                                fetch_k=3
                            )
                            context = " ".join([doc.page_content for doc in context_docs])
                            
                            # Generate response
                            with torch.cuda.amp.autocast():  # Use cuda.amp.autocast() instead
                                try:
                                    response = get_chat_response(llm, user_question, context)
                                    if response and len(response.strip()) > 0:
                                        st.write("Answer:", response)
                                    else:
                                        st.error("No valid response generated")
                                except Exception as e:
                                    st.error(f"Error in response generation: {str(e)}")
                            
                            # Clear GPU memory
                            torch.cuda.empty_cache()
                                            
                        except Exception as e:
                            st.error(f"Error in context retrieval: {str(e)}")
                            
        except Exception as e:
            st.error(f"Error processing document: {str(e)}")

if __name__ == "__main__":
    main()
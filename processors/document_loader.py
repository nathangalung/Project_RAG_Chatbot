from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import tempfile
import os

def load_document(uploaded_file):
    """Load and process a PDF document from Streamlit uploaded file"""
    try:
        # Create a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            # Write uploaded file content to temporary file
            tmp_file.write(uploaded_file.getvalue())
            tmp_path = tmp_file.name
        
        # Load and process the PDF
        loader = PyPDFLoader(tmp_path)
        documents = loader.load()
        
        # Clean up temporary file
        os.unlink(tmp_path)
        
        # Split text into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        
        splits = text_splitter.split_documents(documents)
        return splits
        
    except Exception as e:
        raise Exception(f"Error processing PDF: {str(e)}")
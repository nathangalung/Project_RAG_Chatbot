from langchain_community.embeddings import HuggingFaceEmbeddings

def get_embeddings():
    """Get embeddings model"""
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cuda'}
    )
    return embeddings
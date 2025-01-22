from langchain.prompts import PromptTemplate

def get_qa_template():
    """Get the Q&A prompt template"""
    template = """Use the following context to answer the question. 
    If you don't know the answer, just say you don't know.
    
    Context: {context}
    
    Question: {question}
    
    Answer:"""
    
    return PromptTemplate(
        input_variables=["context", "question"],
        template=template
    )
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch

def get_llm():
    try:
        # Use smaller model more suitable for 8GB VRAM
        model_name = "facebook/opt-125m"
        
        print("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        print("Loading model...")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True
        )

        # Create pipeline with optimized settings
        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=256,
            do_sample=True,
            temperature=0.7,
            top_p=0.95,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            return_full_text=False
        )

        return pipe
    except Exception as e:
        print(f"Error initializing LLM: {str(e)}")
        raise

def get_chat_response(llm, query, context=None):
    try:
        if context:
            prompt = f"""Based on the following context, answer the question.
            If you don't know, just say "I don't know".
            
            Context: {context}
            Question: {query}
            Answer: """
        else:
            prompt = query
            
        # Generate response with proper handling
        outputs = llm(prompt)
        if isinstance(outputs, list) and len(outputs) > 0:
            response = outputs[0].get('generated_text', '').strip()
            return response
        return "I apologize, I could not generate a response."
        
    except Exception as e:
        print(f"Error getting response: {str(e)}")
        return "I encountered an error while processing your request."

def format_sources(source_documents):
    """Format source documents for citation"""
    if not source_documents:
        return ""
    
    formatted_sources = []
    for i, doc in enumerate(source_documents, 1):
        if hasattr(doc, 'metadata') and doc.metadata.get('source'):
            formatted_sources.append(f"{i}. {doc.metadata['source']}")
        else:
            formatted_sources.append(f"{i}. (Source document {i})")
            
    return "\n".join(formatted_sources)
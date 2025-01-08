from langchain_ollama import OllamaEmbeddings

def get_embedding_function():
    """
    Retorna a função de embeddings configurada para o projeto.
    """
    print("🔧 Configurando a função de embeddings com Ollama...")
    # Usa o modelo `nomic-embed-text` via Ollama
    embedding_function = OllamaEmbeddings(model="nomic-embed-text")
    return embedding_function

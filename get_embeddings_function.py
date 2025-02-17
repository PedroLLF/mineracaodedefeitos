# Novo (correto)
from langchain_huggingface import HuggingFaceEmbeddings

def get_embedding_function():
    """
    Retorna a função de embeddings configurada para rodar localmente.
    """
    print("🔧 Configurando a função de embeddings localmente...")

    model_name = "BAAI/bge-large-en-v1.5"  # Certifique-se de que este modelo está disponível localmente
    embedding_function = HuggingFaceEmbeddings(model_name=model_name)

    return embedding_function  # Retorna um objeto compatível com LangChain

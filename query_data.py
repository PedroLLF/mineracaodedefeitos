import pandas as pd
from langchain_chroma import Chroma
from langchain_ollama import OllamaLLM
from get_embeddings_function import get_embedding_function

# Caminho do banco de dados
CHROMA_PATH = "C:/Users/pedro/OneDrive/Documentos/basesdedefeitos/chroma_db"

def main():
    """
    Função principal para executar o fluxo de consulta.
    """
    print("🔍 Consultando o banco de defeitos...")
    query_rag()


def preprocess_csv(file_path):
    """
    Lê o CSV, identifica o defeito mais comum na coluna 'Summary' e retorna um texto formatado.
    """
    # Ler o arquivo CSV
    df = pd.read_csv(file_path)
    
    # Verificar se a coluna 'Summary' existe
    if "Summary" not in df.columns:
        raise KeyError("A coluna 'Summary' não foi encontrada no arquivo CSV. Verifique o formato do arquivo.")
    
    # Encontrar o defeito mais comum
    most_common_defect = df["Summary"].value_counts().idxmax()
    count = df["Summary"].value_counts().max()

    # Retornar o contexto com o defeito mais comum
    return f"O defeito mais comum é: '{most_common_defect}', reportado {count} vezes."


def query_rag():
    """
    Solicita uma pergunta ao usuário e consulta o banco de dados
    para obter a resposta da LLM, com um template de prompt explícito.
    """
    # Template do prompt com instruções claras
    PROMPT_TEMPLATE = """
    Você é um assistente que responde perguntas sobre defeitos no banco de dados do Jira. Responda no seguinte formato:

    "Resposta: [informação específica em português no formato direto]."

    Contexto:
    {contexto}

    Pergunta: {pergunta_do_usuário}
    """

    # Solicitar input da pergunta ao usuário
    query_text = input("Pergunte algo sobre o banco de defeitos: ")

    # Inicializar o banco de dados com embeddings configurados
    embedding_function = get_embedding_function()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    # Buscar os documentos mais relevantes
    print("🔍 Buscando contexto relevante no banco de dados...")
    results = db.similarity_search_with_score(query_text, k=5)

    # Construir o contexto com os documentos relevantes
    context_text = "\n\n---\n\n".join([doc.page_content for doc, _ in results])

    # Adicionar informações do defeito mais comum
    csv_context = preprocess_csv("C:/Users/pedro/OneDrive/Documentos/basesdedefeitos/jirabugs.csv")
    combined_context = f"{csv_context}\n\n---\n\n{context_text}"
    print(f"📄 Contexto enviado para a LLM:\n{combined_context}")

    # Substituir o template do prompt com os valores reais
    prompt = PROMPT_TEMPLATE.format(
        contexto=combined_context,
        pergunta_do_usuário=query_text
    )

    # Consultar a LLM
    print("🤖 Consultando a LLM com o prompt gerado...")
    model = OllamaLLM(model="llama2")
    response_text = model.invoke(prompt).strip()

    # Capturar as fontes do contexto utilizado
    sources = [doc.metadata.get("id", "Desconhecido") for doc, _ in results]
    formatted_response = f"Resposta: {response_text}\nFontes: {sources}"

    # Exibir e retornar a resposta final formatada
    print(formatted_response)
    return formatted_response

if __name__ == "__main__":
    main()

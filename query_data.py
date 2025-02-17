import json
from langchain_chroma import Chroma
from langchain_ollama import OllamaLLM
from get_embeddings_function import get_embedding_function
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, average_precision_score
from rankeval.metrics import MRR, MAP
import re


# Caminho do banco de dados e JSON analítico
CHROMA_PATH = "C:/Users/pedro/OneDrive/Documentos/basesdedefeitos/chroma_db"
JSON_ANALITICO_PATH = "contexto_analitico.json"
INTERACOES_PATH = "interacoes.json"  # Arquivo para salvar as interações

# Ground truth atualizado
ground_truth = {
    "Existem bugs no banco de defeitos jirabugs.csv que mencionam CTO? Quais áreas do sistema podem estar impactadas?": {"keywords": ["CTO"], "relevant_keys": ["CTO"]},
    "O banco jirabugs.csv contém registros de falhas envolvendo splitter? Quais setores do sistema são afetados?": {"keywords": ["splitter"], "relevant_keys": ["splitter"]},
    "Foram reportados problemas relacionados a DIO no arquivo jirabugs.csv? Que impacto esses bugs podem ter no sistema?": {"keywords": ["DIO"], "relevant_keys": ["DIO"]},
    "Houve registros de defeitos envolvendo cabos no banco de defeitos? Em quais partes do sistema esses problemas se manifestam?": {"keywords": ["cabo"], "relevant_keys": ["cabo"]},
    "O banco jirabugs.csv possui relatos de falhas ligadas à OLT? Existe alguma área do sistema mais suscetível a esses problemas?": {"keywords": ["OLT"], "relevant_keys": ["OLT"]},
    "Existe algum padrão nos defeitos relacionados a splitter, CTO e DIO no banco jirabugs.csv? Como essas falhas se relacionam entre si dentro do sistema?": {"keywords": ["splitter", "CTO", "DIO"], "relevant_keys": ["splitter", "CTO", "DIO"]},
    "Quais falhas foram registradas no Epic Link BR-27? Esses problemas compartilham alguma característica comum?": {"keywords": ["BR-27"], "relevant_keys": ["BR-27"]},
    "Há bugs relacionados a CEO no banco de defeitos? Que partes do sistema são mais impactadas por essas falhas?": {"keywords": ["CEO"], "relevant_keys": ["CEO"]},
    "Foram identificados problemas no sistema relacionados a MAC? Qual a frequência desses registros no banco jirabugs.csv?": {"keywords": ["MAC"], "relevant_keys": ["MAC"]},
    "O termo 'Uplink' aparece em relatórios de falhas no banco jirabugs.csv? Há um padrão entre esses registros?": {"keywords": ["Uplink"], "relevant_keys": ["Uplink"]},
    "Que problemas foram identificados na versão [v1.5.0-sp27.2] no banco jirabugs.csv? Essas falhas são recorrentes em versões anteriores?": {"keywords": ["v1.5.0-sp27.2"], "relevant_keys": ["v1.5.0-sp27.2"]},
    "Há registros de bugs relacionados ao mapa no banco jirabugs.csv? Que impacto esses problemas podem ter na funcionalidade do sistema?": {"keywords": ["mapa"], "relevant_keys": ["mapa"]},
    "Os registros do banco jirabugs.csv mencionam falhas associadas a FK-CTOP? Essas falhas têm relação com outros componentes do sistema?": {"keywords": ["FK-CTOP"], "relevant_keys": ["FK-CTOP"]},
    "Quais bugs foram identificados na versão [v1.5.0-sp29.1]? Essas falhas são específicas desta versão ou aparecem em outras releases?": {"keywords": ["v1.5.0-sp29.1"], "relevant_keys": ["v1.5.0-sp29.1"]},
    "O banco jirabugs.csv contém reports atribuídos a Ismayle Santos? Quais tipos de problemas ele reportou?": {"keywords": ["Ismayle"], "relevant_keys": ["Ismayle"]},
    "O erro 'Erro desconhecido' foi registrado no sistema? Em quais situações essa mensagem aparece?": {"keywords": ["Erro desconhecido"], "relevant_keys": ["Erro desconhecido"]},
    "Existem registros de falhas no banco jirabugs.csv que mencionam KMZ? Que tipo de problema essas ocorrências indicam?": {"keywords": ["KMZ"], "relevant_keys": ["KMZ"]},
    "Quais falhas estão associadas às chaves BR-3054, BR-3046, BR-3042, BR-3041, BR-3027, BR-3018, BR-3017, BR-3013, BR-3010 e BR-3009? Esses reports indicam alguma relação entre os defeitos?": {"keywords": ["BR-3054", "BR-3046", "BR-3042", "BR-3041", "BR-3027", "BR-3018", "BR-3017", "BR-3013", "BR-3010", "BR-3009"], "relevant_keys": ["BR-3054", "BR-3046", "BR-3042", "BR-3041", "BR-3027", "BR-3018", "BR-3017", "BR-3013", "BR-3010", "BR-3009"]},
    "Os registros no banco jirabugs.csv indicam problemas com o botão 'Salvar'? Em quais contextos esse problema ocorre?": {"keywords": ["Salvar"], "relevant_keys": ["Salvar"]},
    "Quais são os principais defeitos relacionados a Endereços? Essas falhas impactam a usabilidade do sistema?": {"keywords": ["Endereço"], "relevant_keys": ["Endereço"]}
}

LIMIAR_RELEVANCIA = 0.50  # 50% dos documentos devem conter a palavra-chave para ser considerado relevante

def convert_to_binary(retrieved_docs, relevant_keys):
    """
    Converte retrieved_docs e relevant_keys em listas binárias.
    
    - retrieved_docs: Lista de documentos recuperados.
    - relevant_keys: Lista de palavras-chave relevantes.

    Retorna:
    - y_true: Lista binária indicando se cada documento recuperado é relevante (1 = relevante, 0 = não relevante).
    - y_pred: Lista binária indicando se cada documento foi recuperado (1 = recuperado, 0 = não recuperado).
    """
    relevant_keys = [key.lower() for key in relevant_keys]  # Normaliza as palavras-chave

    y_true = [1 if any(key in doc.lower() for key in relevant_keys) else 0 for doc in retrieved_docs]
    y_pred = [1] * len(retrieved_docs)  # Todos os documentos recuperados são considerados 1

    return y_true, y_pred

def precision_at_k(y_true, y_pred, k):
    """
    Calcula a precisão no topo k dos documentos recuperados.

    Retorna:
    - Precisão no topo k.
    """
    k = max(1, k)  # Evita divisão por zero
    return precision_score(y_true[:k], y_pred[:k], zero_division=0)

def recall_at_k(y_true, y_pred, k):
    """
    Calcula o recall no topo k dos documentos recuperados.

    Retorna:
    - Recall no topo k.
    """
    k = max(1, k)
    return recall_score(y_true[:k], y_pred[:k], zero_division=0)

def f1_score_at_k(y_true, y_pred, k):
    """
    Calcula o F1-score no topo k dos documentos recuperados usando sklearn.metrics.f1_score.

    Retorna:
    - F1-score no topo k.
    """
    k = max(1, k)
    return f1_score(y_true[:k], y_pred[:k], zero_division=0)

def average_precision(y_true, y_pred):
    """
    Calcula a precisão média (Average Precision, AP) usando sklearn.metrics.average_precision_score.

    Retorna:
    - Precisão média (AP).
    """
    return average_precision_score(y_true, y_pred)


def calculate_relevance(retrieved_docs, keyword):
    """Calcula a relevância verificando quantos documentos contêm a palavra-chave."""
    relevant_count = sum(1 for doc in retrieved_docs if keyword.lower() in doc.lower())
    relevance_ratio = relevant_count / len(retrieved_docs) if retrieved_docs else 0
    return relevance_ratio

def salvar_interacao(query, documentos, resposta, metricas):
    """Salva a interação em um arquivo JSON."""
    try:
        with open(INTERACOES_PATH, 'r', encoding='utf-8') as f:
            interacoes = json.load(f)
    except FileNotFoundError:
        interacoes = []
    
    interacoes.append({
        "query": query,
        "documentos": documentos,
        "resposta": resposta,
        "metricas": metricas
    })
    
    with open(INTERACOES_PATH, 'w', encoding='utf-8') as f:
        json.dump(interacoes, f, ensure_ascii=False, indent=4)

def filter_relevant_docs(retrieved_docs, keywords):
    """Filtra os documentos recuperados que contêm pelo menos uma das palavras-chave."""
    return [doc for doc in retrieved_docs if any(keyword.lower() in doc.lower() for keyword in keywords)]

        
def calcular_metricas(retrieved_docs, keywords, relevant_keys, k=2):
    """Calcula todas as métricas de avaliação."""
    relevant_docs = filter_relevant_docs(retrieved_docs, keywords)
    total_non_relevant = len(retrieved_docs) - len(relevant_docs)

    # Converter para listas binárias
    y_true, y_pred = convert_to_binary(retrieved_docs, relevant_keys)

    return {
        "precision_at_k": precision_at_k(y_true, y_pred, k),
        "recall_at_k": recall_at_k(y_true, y_pred, k),
        "f1_score": f1_score_at_k(y_true, y_pred, k),
        "average_precision": average_precision(y_true, y_pred),
    }        
        

def query_data():
    PROMPT_TEMPLATE = """
    You are an assistant specialized in answering questions about defects in a GPON (Gigabit Passive Optical Network) management system. Your task is to analyze the provided context and generate accurate, concise, and relevant responses in Portuguese.

    ### Context:
    You are working with a Jira defect report database (jirabugs.csv) related to a web-based GPON network management system. The system manages components such as:

    CTO (Optical Termination Box): Distributes optical fibers to multiple customers.
    CEO (Optical Splice Closure): Protects and organizes fiber splices.
    DIO (Optical Distribution Frame): Concentrates and distributes fibers in indoor environments.
    OLT (Optical Line Terminal): The central GPON network equipment that connects the optical infrastructure to the service provider.
    ONU (Optical Network Unit): The customer-end device that receives the optical signal from the OLT.
    Splitter: Divides a single optical signal into multiple signals for efficient network distribution.
    Optical Cable: The physical medium that carries fiber optic signals between network elements.
    The defects recorded in the database may relate to functionality failures, user interface issues, data handling errors, system performance problems, and other critical aspects of GPON network management.

    ### Data Structure:
    The defect reports are organized in a CSV file with the following columns:
    1. **Issue Type**: The type of issue (e.g., "Dev Bug", "Bug", "Task"). This indicates the nature of the defect.
    2. **Key**: A unique identifier for the defect (e.g., "BR-3054"). Use this to reference specific defects.
    3. **Summary**: A brief description of the defect. This provides details about the issue.
    4. **Status**: The current status of the defect (e.g., "To Do", "In Progress", "Done"). This indicates the progress in resolving the issue.
    5. **Created**: The date and time when the defect was reported.
    6. **Linked Issues**: A list of related defect keys (e.g., "BR-2335, BR-2946"). These indicate dependencies or related issues.
    7. **Development**: Additional development-related information (usually empty or contains metadata).
    8. **Epic Link**: The key of the Epic to which the defect belongs (e.g., "BR-897"). Epics group related defects.
    9. **Reporter**: The name of the person who reported the defect.
    10. **Epic Name**: The name of the Epic (usually empty in the data).
    11. **Sprint**: The Sprint in which the defect is being addressed (e.g., "BR Sprint 37"). Sprints are time-boxed development cycles.

    ### Instructions:
    1. **Understand the Query**:Carefully analyze the query to identify the GPON network components, system functionalities or system areas being questioned..
    2. **Use the Context**: The defect reports may provide insights into recurring failures and critical areas. 
    3. **Retrieved Documents**: Below the context, you will find a list of retrieved documents (defect reports) that are relevant to the query. Use these documents to provide specific details in your response.
    4. **Critical analysis and Inference**: Try to identify hidden patterns based on retrieved defects. 
       - Example: If multiple bugs mention CTO and connection failures, this may indicate a structural issue in CTO management.
       - Relate defects to potential system-critical areas (e.g., ONU provisioning interface, OLT integration, splitter registration stability).
    5. **Response Format**: Your response should be in Portuguese and follow this format:
       - Start with "Resposta:" followed by a direct and specific answer to the query.
       - Include relevant details from the retrieved documents, such as summaries, and affected areas.
       - If possible, infer systemic problems or recurring trends.
       - Conclude by listing the sources (defect keys) used in the response.

    ### Example:
    Query: "Quais os bugs relacionados a CTO no banco de defeitos jirabugs.csv? Qual a possível área do sistema afetada?"
    Resposta: Os bugs relacionados a CTO incluem problemas de conexão com cabos de distribuição, instanciação incorreta e exibição de mensagens de erro. A área do sistema afetada é a funcionalidade de gerenciamento de CTOs, especificamente a interface de conexão e instanciação. 
    Fontes: BR-2199, BR-2200, BR-1773.

    ### Data for Analysis:
    - Context: {contexto}
    - Query: {pergunta_do_usuário}
    """

    query_text = input("❓ Sua pergunta: ")
    
    embedding_function = get_embedding_function()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    print("🔍 Consultando o banco de dados Chroma para contexto adicional...")
    results = db.similarity_search_with_score(query_text, k=20)

    retrieved_docs = [doc.page_content for doc, _ in results]  # Documentos recuperados

    print("📊 Carregando contexto analítico do JSON...")
    with open(JSON_ANALITICO_PATH, 'r', encoding='utf-8') as arquivo_json:
        contexto_analitico = json.load(arquivo_json)

    contexto_analitico_texto = (
        f"Contagem total de bugs: {contexto_analitico['contagem_total_bugs']}\n\n"
        f"Contagem de bugs por Epic Link:\n{contexto_analitico['contagem_epic_link']}\n\n"
        f"Contagem de bugs por release:\n{contexto_analitico['contagem_bugs_release']}"
    )

    combined_context = f"{contexto_analitico_texto}\n\n---\n\n" + "\n\n---\n\n".join(retrieved_docs)
    print(f"📄 Contexto combinado:\n{combined_context}")

    prompt = PROMPT_TEMPLATE.format(contexto=combined_context, pergunta_do_usuário=query_text)

    print("🤖 Consultando a LLM com o prompt gerado...")
    model = OllamaLLM(model="mistral")
    response = model.invoke(prompt).strip()
    
    sources = [doc.metadata.get("id", "Desconhecido") for doc, _ in results]
    formatted_response = f"💡 Resposta: {response}\n🔗 Fontes: {sources}"

    # Calcular métricas
    if query_text in ground_truth:
        keywords = ground_truth[query_text]["keywords"]
        relevant_keys = ground_truth[query_text]["relevant_keys"]
        relevant_docs = filter_relevant_docs(retrieved_docs, keywords)  # Filtra documentos relevantes
        total_non_relevant = len(retrieved_docs) - len(relevant_docs)
        k = 4  # Definir o K para avaliação

        # Converter para listas binárias
        y_true, y_pred = convert_to_binary(retrieved_docs, relevant_keys)

        metricas = {
            "precision_at_k": precision_at_k(y_true, y_pred, k),
            "recall_at_k": recall_at_k(y_true, y_pred, k),
            "f1_score": f1_score_at_k(y_true, y_pred, k),
            "average_precision": average_precision(y_true, y_pred),
        }

        # Salvar interação com métricas
        salvar_interacao(query_text, retrieved_docs, response, metricas)
            
    print(formatted_response)
    return formatted_response

if __name__ == "__main__":
    query_data()
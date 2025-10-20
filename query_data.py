import json
from langchain_chroma import Chroma
from langchain_ollama import OllamaLLM
from get_embeddings_function import get_embedding_function
from sklearn.metrics import precision_score, recall_score, f1_score, average_precision_score

# Caminhos dos arquivos
CHROMA_PATH = "C:/Users/pedro/OneDrive/Documentos/basesdedefeitos/chroma_db"
JSON_ANALITICO_PATH = "contexto_analitico.json"
INTERACOES_PATH = "interacoes.json"

# === Ground truth completo e preservado ===
ground_truth = {
    "Quais os bugs relacionados a CTO? Quais áreas do sistema podem estar impactadas?": {"keywords": ["CTO"], "relevant_keys": ["CTO"]},
    "O banco de defeitos contém registros de falhas envolvendo splitter? Quais setores do sistema são afetados?": {"keywords": ["splitter"], "relevant_keys": ["splitter"]},
    "O banco jirabugs.csv possui relatos de falhas ligadas à OLT? Existe alguma área do sistema mais suscetível a esses problemas?": {"keywords": ["OLT"], "relevant_keys": ["OLT"]},
    "Houve registros de defeitos envolvendo cabos no banco de defeitos? Como podemos apontar possíveis áreas críticas do sistema a partir disso?": {"keywords": ["cabo"], "relevant_keys": ["cabo"]},
    "Existe algum padrão nos defeitos relacionados a splitter, CTO e CEO no banco jirabugs.csv? Como essas falhas se relacionam entre si dentro do sistema?": {"keywords": ["CTO", "CEO", "Splitter"], "relevant_keys": ["CTO", "CEO", "Splitter"]},
    "Foram identificados problemas no sistema relacionados a MAC?": {"keywords": ["MAC"], "relevant_keys": ["MAC"]},
    "O termo 'Uplink' aparece em relatórios de falhas no banco jirabugs.csv?": {"keywords": ["Uplink"], "relevant_keys": ["Uplink"]},
    "Há registros de bugs relacionados a 'mapa' no banco jirabugs.csv?": {"keywords": ["mapa"], "relevant_keys": ["mapa"]},
    "Os registros no banco jirabugs.csv indicam problemas com o botão 'Salvar'? Em quais contextos esse problema ocorre?": {"keywords": ["Salvar"], "relevant_keys": ["Salvar"]},
    "'Erro desconhecido' foi registrado no sistema? Em quais situações essa mensagem aparece?": {"keywords": ["Erro desconhecido"], "relevant_keys": ["Erro desconhecido"]},
    "Quais os reports de Epic Link BR-27": {"keywords": ["BR-27"], "relevant_keys": ["BR-27"]},
    "O que foi reportado na Key BR-2015?": {"keywords": ["Splitter", "CTO"], "relevant_keys": ["splitter", "CTO"]},
    "Pesquise os reports feitos na Sprint 32 e faça um relatório sobre esses reports": {"keywords": ["Sprint 32"], "relevant_keys": ["Sprint 32"]},
    "Quais relatos foram feitos pelo reporter Ismayle Santos? Quais tipos de problemas ele reportou?": {"keywords": ["Ismayle"], "relevant_keys": ["Ismayle"]},
    "Analise os problemas associados a linked issues BR-1154. Esses defeitos possuem um padrão?": {"keywords": ["BR-1154"], "relevant_keys": ["BR-1154"]},
    "Qual a cor do cavalo branco de napoleão?": {"keywords": ["cavalo branco"], "relevant_keys": ["cavalo branco"]},
    "Com quantos paus se faz uma canoa?": {"keywords": ["canoa"], "relevant_keys": ["canoa"]},
    "Quanto é 10+10?": {"keywords": ["10"], "relevant_keys": ["10"]},
    "Qual o esporte mais praticado no mundo?": {"keywords": ["esporte"], "relevant_keys": ["esporte"]},
    "Indique um catálogo de filmes lançados em 2024": {"keywords": ["filmes"], "relevant_keys": ["filmes"]},

    # English translations
    "What are the bugs related to CTO? Which areas of the system might be impacted?": {"keywords": ["CTO"], "relevant_keys": ["CTO"]},
    "Does the defect database contain records of failures involving splitters? Which sectors of the system are affected?": {"keywords": ["splitter"], "relevant_keys": ["splitter"]},
    "Does the jirabugs.csv database contain reports of failures related to OLT? Is there any area of the system more susceptible to these issues?": {"keywords": ["OLT"], "relevant_keys": ["OLT"]},
    "Were there records of defects involving cables in the defect database? How can we identify potential critical areas of the system from this?": {"keywords": ["cabo"], "relevant_keys": ["cabo"]},
    "Is there any pattern in the defects related to splitters, CTO, and CEO in the jirabugs.csv database? How do these failures relate to each other within the system?": {"keywords": ["CTO", "CEO", "Splitter"], "relevant_keys": ["CTO", "CEO", "Splitter"]},
    "Were any issues related to MAC identified in the system?": {"keywords": ["MAC"], "relevant_keys": ["MAC"]},
    "Does the term 'Uplink' appear in failure reports in the jirabugs.csv database?": {"keywords": ["Uplink"], "relevant_keys": ["Uplink"]},
    "Are there records of bugs related to 'map' in the jirabugs.csv database?": {"keywords": ["map"], "relevant_keys": ["map"]},
    "Do the records in the jirabugs.csv database indicate issues with the 'Save' button? In what contexts does this problem occur?": {"keywords": ["Salva"], "relevant_keys": ["Salva"]},
    "Was 'Unknown error' recorded in the system? In what situations does this message appear?": {"keywords": ["Erro Desconhecido"], "relevant_keys": ["Erro Desconhecido"]},
    "What are the reports for Epic Link BR-27?": {"keywords": ["BR-27"], "relevant_keys": ["BR-27"]},
    "What was reported in Key BR-2015?": {"keywords": ["Splitter", "CTO"], "relevant_keys": ["splitter", "CTO"]},
    "Search for reports made in Sprint 32 and create a report on these issues.": {"keywords": ["Sprint 32"], "relevant_keys": ["Sprint 32"]},
    "What reports were made by the reporter Ismayle Santos? What types of issues did he report?": {"keywords": ["Ismayle"], "relevant_keys": ["Ismayle"]},
    "Analyze the issues associated with linked issues BR-1154. Do these defects have a pattern?": {"keywords": ["BR-1154"], "relevant_keys": ["BR-1154"]},
    "What is the color of Napoleon's white horse?": {"keywords": ["white horse"], "relevant_keys": ["white horse"]},
    "How many sticks does it take to make a canoe?": {"keywords": ["canoe"], "relevant_keys": ["canoe"]},
    "What is 10+10?": {"keywords": ["10"], "relevant_keys": ["10"]},
    "What is the most practiced sport in the world?": {"keywords": ["sport"], "relevant_keys": ["sport"]},
    "Provide a catalog of movies released in 2024.": {"keywords": ["movies"], "relevant_keys": ["movies"]}
}


# --- Funções auxiliares ---

def convert_to_binary(retrieved_docs, relevant_keys):
    """Converte documentos recuperados em listas binárias para cálculo de métricas."""
    relevant_keys = [key.lower() for key in relevant_keys]
    y_true = [1 if any(key in doc.lower() for key in relevant_keys) else 0 for doc in retrieved_docs]
    y_pred = [1] * len(retrieved_docs)
    return y_true, y_pred


def calcular_metricas(retrieved_docs, relevant_keys):
    """Calcula métricas de avaliação com sklearn."""
    y_true, y_pred = convert_to_binary(retrieved_docs, relevant_keys)
    return {
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1_score": f1_score(y_true, y_pred, zero_division=0),
        "average_precision": average_precision_score(y_true, y_pred),
    }


def salvar_interacao(query, documentos, resposta, metricas):
    """Salva interações no JSON."""
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


# --- Função principal ---
def query_data():
    """Executa a consulta e calcula métricas."""
    PROMPT_TEMPLATE = """
    Você é um assistente especializado em responder perguntas sobre defeitos em sistemas GPON.
    Analise o contexto e gere respostas concisas em português.

    ### Contexto:
    {contexto}

    ### Pergunta:
    {pergunta_do_usuário}

    Responda no formato:
    Resposta: <texto direto e objetivo>
    Fontes: <lista de chaves de defeito usadas>
    """

    query_text = input("❓ Sua pergunta: ")
    embedding_function = get_embedding_function()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)
    results = db.similarity_search_with_score(query_text, k=20)
    retrieved_docs = [doc.page_content for doc, _ in results]

    with open(JSON_ANALITICO_PATH, 'r', encoding='utf-8') as f:
        contexto_analitico = json.load(f)

    contexto_texto = (
        f"Total de bugs: {contexto_analitico['contagem_total_bugs']}\n\n"
        f"Bugs por Epic Link:\n{contexto_analitico['contagem_epic_link']}\n\n"
        f"Bugs por Release:\n{contexto_analitico['contagem_bugs_release']}"
    )

    combined_context = f"{contexto_texto}\n\n---\n\n" + "\n\n---\n\n".join(retrieved_docs)
    prompt = PROMPT_TEMPLATE.format(contexto=combined_context, pergunta_do_usuário=query_text)

    model = OllamaLLM(model="mistral")
    response = model.invoke(prompt).strip()
    sources = [doc.metadata.get("id", "Desconhecido") for doc, _ in results]

    # Calcular métricas com sklearn
    metricas = {}
    if query_text in ground_truth:
        relevant_keys = ground_truth[query_text]["relevant_keys"]
        metricas = calcular_metricas(retrieved_docs, relevant_keys)
        salvar_interacao(query_text, retrieved_docs, response, metricas)

    print(f"\n💡 Resposta: {response}\n🔗 Fontes: {sources}\n📈 Métricas: {metricas}")
    return response


if __name__ == "__main__":
    query_data()

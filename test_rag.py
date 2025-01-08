from query_data import query_rag
from langchain_ollama import OllamaLLM

# Prompt usado para validar respostas
EVAL_PROMPT = """
Você é um validador. Compare a resposta esperada com a resposta fornecida:

Resposta esperada:
{expected_response}

Resposta fornecida:
{actual_response}

A resposta fornecida é exatamente igual à resposta esperada? Responda apenas com "verdadeiro" ou "falso".
"""

def query_rag_with_arg(question: str):
    """
    Adapta a função query_rag para aceitar uma pergunta como argumento.
    """
    # Salvar o comportamento original da função
    original_input = __builtins__.input
    __builtins__.input = lambda _: question

    # Executar query_rag simulando o input
    try:
        response = query_rag()
    finally:
        # Restaurar o comportamento original do input
        __builtins__.input = original_input

    return response


def test_sample_query():
    """
    Testa um exemplo de consulta com base no contexto do CSV.
    """
    assert query_and_validate(
        question="Qual o Epic Link que mais aparece no banco de defeitos jirabugs.csv e quantas vezes ele aparece?",
        expected_response="Resposta: O Epic Link 'BR-27' é o que mais aparece, com 117 ocorrências.\nFontes: [lista de fontes]."
    )


def query_and_validate(question: str, expected_response: str):
    """
    Faz a consulta e valida a resposta com base no esperado.
    """
    print(f"❓ Pergunta: {question}")
    # Obtém resposta da RAG
    response_text = query_rag_with_arg(question)

    # Formata o prompt de validação
    prompt = EVAL_PROMPT.format(
        expected_response=expected_response, actual_response=response_text
    )
    print("🤔 Validando resposta...")

    # Consulta a LLM para validação
    model = OllamaLLM(model="llama2")
    evaluation_results = model.invoke(prompt).strip().lower()

    if "verdadeiro" in evaluation_results:
        print("\033[92m✔ Resposta validada com sucesso!\033[0m")
        return True
    elif "falso" in evaluation_results:
        print("\033[91m✘ Resposta incorreta!\033[0m")
        return False
    else:
        raise ValueError(
            f"Resultado inválido na validação: {evaluation_results}"
        )


if __name__ == "__main__":
    # Executar testes manuais
    test_sample_query()

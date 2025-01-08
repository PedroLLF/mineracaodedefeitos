from langchain_ollama import OllamaLLM
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, END, StateGraph, MessagesState
from langgraph.prebuilt import ToolNode

# Ferramenta para interagir com DataFrames do pandas
import pandas as pd
import uuid  # Para gerar um UUID único para o thread_id

# Carrega o arquivo CSV (ajuste o caminho do arquivo)
df = pd.read_csv("C:/Users/pedro/OneDrive/Documentos/basesdedefeitos/jirabugs.csv")

@tool
def analyze_epics(query: str):
    """
    Ferramenta para analisar a frequência dos épicos no DataFrame pandas.
    """
    if "frequência de épicos" in query.lower():
        # Conta a frequência de cada épico
        epic_counts = df["Epic Link"].value_counts()
        return f"A frequência de ocorrência por épico é: {epic_counts.to_dict()}"
    elif "épico mais comum" in query.lower():
        # Identifica o épico mais frequente
        most_common_epic = df["Epic Link"].value_counts().idxmax()
        count = df["Epic Link"].value_counts().max()
        return f"O épico mais comum é '{most_common_epic}' com {count} ocorrências."
    else:
        return "Desculpe, não consegui entender a pergunta. Tente ser mais específico sobre a análise de épicos."

# Lista de ferramentas disponíveis
tools = [analyze_epics]

# Define o modelo da LLM (OllamaLLM)
llm = OllamaLLM(model="llama2")

# Integra ferramentas no fluxo (usando ToolNode)
tool_node = ToolNode(tools)

# Função para decidir o próximo passo no fluxo
def should_continue(state: MessagesState):
    """
    Decide se o fluxo deve continuar ou terminar, com base na resposta.
    """
    messages = state["messages"]
    last_message = messages[-1]
    if last_message.tool_calls:
        return "tools"  # Rota para usar as ferramentas
    return END  # Termina o fluxo

# Função que chama a LLM para responder
def call_model(state: MessagesState):
    """
    Chama o modelo LLM para gerar respostas.
    """
    messages = state["messages"]
    response = llm.invoke(messages)
    return {"messages": [response]}

# Cria o gráfico do LangGraph para fluxo do agente
workflow = StateGraph(MessagesState)

# Adiciona os nós ao gráfico
workflow.add_node("agent", call_model)  # Nó principal da LLM
workflow.add_node("tools", tool_node)  # Nó das ferramentas

# Define o ponto de entrada (primeiro nó a ser executado)
workflow.add_edge(START, "agent")

# Adiciona condições para alternar entre nós
workflow.add_conditional_edges("agent", should_continue)
workflow.add_edge("tools", "agent")

# Inicializa a memória para salvar o estado
checkpointer = MemorySaver(thread_id=str(uuid.uuid4()))  # Adicionando thread_id gerado dinamicamente

# Compila o fluxo para ser usado
app = workflow.compile(checkpointer=checkpointer)

# Executa o fluxo
if __name__ == "__main__":
    print("❓ Pergunta: Qual é a frequência de ocorrência por épico?")
    initial_state = {"messages": [HumanMessage(content="Qual é a frequência de ocorrência por épico?")]}
    final_state = app.invoke(initial_state)
    print(f"🤖 Resposta: {final_state['messages'][-1].content}")

import argparse
import os
import shutil
import pandas as pd
from langchain.schema import Document
from langchain.vectorstores import Chroma
from get_embeddings_function import get_embedding_function
from langchain.text_splitter import CharacterTextSplitter
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain.chains.query_constructor.base import AttributeInfo

# Caminhos do banco de dados e dos dados
CHROMA_PATH = "C:/Users/pedro/OneDrive/Documentos/basesdedefeitos/chroma_db"
DATA_PATH = "C:/Users/pedro/OneDrive/Documentos/basesdedefeitos/jirabugs.csv"

# Definição dos metadados utilizados para a filtragem
METADATA_FIELD_INFO = [
    AttributeInfo(name="id", description="O identificador único do bug", type="string"),
    AttributeInfo(name="status", description="O status atual do bug (ex: Open, Closed)", type="string"),
    AttributeInfo(name="priority", description="A prioridade do bug (ex: High, Medium, Low)", type="string"),
    AttributeInfo(name="assignee", description="A pessoa responsável pelo bug", type="string"),
]

def main():
    """
    Configura o pipeline para popular o banco de dados Chroma.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true", help="Resetar o banco de dados.")
    args = parser.parse_args()

    if args.reset:
        print("✨ Limpando o banco de dados...")
        clear_database()

    print("📥 Carregando documentos do CSV...")
    documents = load_documents_from_csv()
    print(f"📝 Total de documentos carregados: {len(documents)}")

    print("➖ Dividindo documentos em chunks...")
    chunks = split_documents(documents)

    print("➕ Adicionando chunks ao banco de dados...")
    add_to_chroma(chunks)

def load_documents_from_csv():
    """
    Carrega documentos do CSV, aplicando pré-processamento e enriquecimento de metadados.
    """
    print("📥 Carregando documentos do CSV...")
    df = pd.read_csv(DATA_PATH)

    documents = []
    for _, row in df.iterrows():
        # Concatena os campos relevantes em uma única string para o page_content
        content = f"Summary: {row['Summary'] if pd.notna(row['Summary']) else 'No summary available'}\n" \
                  f"Epic Link: {row['Epic Link'] if pd.notna(row['Epic Link']) else 'No Epic Link available'}\n" \
                  f"Reporter: {row['Reporter'] if pd.notna(row['Reporter']) else 'No reporter found'}"

        # Todos os outros campos são metadados
        metadata = {
            "id": row["Key"],
            "status": row["Status"] if "Status" in df.columns else "Unknown",
            "created": row["Created"] if "Created" in df.columns else "Unknown",
            "linked_issues": row["Linked Issues"] if "Linked Issues" in df.columns else "None",
            "development": row["Development"] if "Development" in df.columns else "None",
            "epic_link": row["Epic Link"] if "Epic Link" in df.columns else "None",
            "reporter": row["Reporter"] if "Reporter" in df.columns else "Unknown",
            "epic_name": row["Epic Name"] if "Epic Name" in df.columns else "Unknown",
            "sprint": row["Sprint"] if "Sprint" in df.columns else "None",
        }

        document = Document(
            page_content=content,  # Agora é uma string
            metadata=metadata
        )

        documents.append(document)

    print(f"📝 Total de documentos carregados: {len(documents)}")
    return documents


def split_documents(documents):
    """
    Divide documentos grandes em chunks menores.
    """
    return documents  # Retorna os documentos diretamente

def add_to_chroma(chunks):
    """
    Adiciona ou atualiza chunks no banco de dados Chroma.
    """
    embedding_function = get_embedding_function()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    # Obtém IDs existentes no banco de dados
    existing_items = db.get(include=[])  # IDs são incluídos por padrão
    existing_ids = set(existing_items["ids"])
    print(f"📊 Documentos existentes no banco de dados: {len(existing_ids)}")

    # Adiciona apenas novos chunks
    new_chunks = [chunk for chunk in chunks if chunk.metadata["id"] not in existing_ids]
    if new_chunks:
        print(f"➕ Adicionando {len(new_chunks)} novos documentos...")
        new_chunk_ids = [chunk.metadata["id"] for chunk in new_chunks]
        db.add_documents(new_chunks, ids=new_chunk_ids)
    else:
        print("✅ Nenhum documento novo para adicionar.")

def get_self_query_retriever():
    """
    Configura um Self-Query Retriever para o ChromaDB, permitindo consultas baseadas em metadados.
    """
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=get_embedding_function())

    retriever = SelfQueryRetriever.from_llm(
        llm=None,  # Aqui você pode substituir pelo seu modelo de LLM (ex: OpenAI, Ollama, etc.)
        vectorstore=db,
        document_content_description="Registros de bugs de um sistema",
        metadata_field_info=METADATA_FIELD_INFO,
        search_kwargs={"k": 5},  # Número de documentos retornados
    )

    return retriever

def clear_database():
    """
    Remove todos os dados persistidos no banco de dados Chroma.
    """
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)
        print("✅ Banco de dados resetado com sucesso.")
    else:
        print("✅ Nenhum banco de dados existente para resetar.")

if __name__ == "__main__":
    main()

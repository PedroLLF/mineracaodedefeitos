import argparse
import os
import shutil
import pandas as pd
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_chroma import Chroma
from langchain.docstore.document import Document
from get_embeddings_function import get_embedding_function

# Caminhos do banco de dados e dos dados
CHROMA_PATH = "C:/Users/pedro/OneDrive/Documentos/basesdedefeitos/chroma_db"
DATA_PATH = "C:/Users/pedro/OneDrive/Documentos/basesdedefeitos/jirabugs.csv"  # Nome do arquivo CSV

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
    print(f"📄 Total de documentos carregados: {len(documents)}")

    print("✂ Dividindo documentos em chunks...")
    chunks = split_documents(documents)

    print("➕ Adicionando chunks ao banco de dados...")
    add_to_chroma(chunks)


def load_documents_from_csv():
    print("📥 Carregando documentos do CSV...")
    df = pd.read_csv(DATA_PATH)

    # Ajustar conforme as colunas do CSV
    documents = []
    for _, row in df.iterrows():
        document = Document(
            page_content=row["Summary"],  # Campo principal de texto
            metadata={
                "id": row["Key"],
                "issue_type": row["Issue Type"],
                "status": row["Status"],
                "created": row["Created"],
                "reporter": row["Reporter"],
                # Adicione outros campos conforme necessário
            },
        )
        documents.append(document)

    return documents



def split_documents(documents):
    """
    Divide documentos grandes em chunks menores usando o divisor de texto.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=80,
        length_function=len
    )
    return text_splitter.split_documents(documents)


def add_to_chroma(chunks):
    """
    Adiciona ou atualiza chunks no banco de dados Chroma.
    """
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=get_embedding_function())

    # Calcula IDs únicos para os chunks
    chunks_with_ids = calculate_chunk_ids(chunks)

    # Obtém IDs existentes no banco de dados
    existing_items = db.get(include=[])  # IDs são incluídos por padrão
    existing_ids = set(existing_items["ids"])
    print(f"📊 Documentos existentes no banco de dados: {len(existing_ids)}")

    # Adiciona apenas novos chunks
    new_chunks = [chunk for chunk in chunks_with_ids if chunk.metadata["id"] not in existing_ids]
    if new_chunks:
        print(f"➕ Adicionando {len(new_chunks)} novos documentos...")
        new_chunk_ids = [chunk.metadata["id"] for chunk in new_chunks]
        db.add_documents(new_chunks, ids=new_chunk_ids)
    else:
        print("✅ Nenhum documento novo para adicionar.")



def calculate_chunk_ids(chunks):
    """
    Calcula IDs únicos para cada chunk com base nos metadados.
    """
    for chunk in chunks:
        source = chunk.metadata.get("source")
        doc_id = chunk.metadata.get("id")
        chunk_index = chunk.metadata.get("chunk_index", 0)

        # Gera um ID no formato: `source:doc_id:index`
        chunk_id = f"{source}:{doc_id}:{chunk_index}"
        chunk.metadata["id"] = chunk_id

    return chunks


def clear_database():
    """
    Remove todos os dados persistidos no banco de dados Chroma.
    """
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)


if __name__ == "__main__":
    main()
    
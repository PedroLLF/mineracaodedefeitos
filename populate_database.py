import argparse
import os
import shutil
import pandas as pd
from langchain.schema import Document
from langchain_chroma import Chroma
from get_embeddings_function import get_embedding_function
from langchain.text_splitter import CharacterTextSplitter

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

    # Inicializa o splitter de texto (opcional, para textos longos)
    text_splitter = CharacterTextSplitter(
        separator=" ",
        chunk_size=500,  # Tamanho máximo de cada chunk
        chunk_overlap=50,  # Sobreposição entre chunks
    )

    documents = []
    for _, row in df.iterrows():
        # Conteúdo principal: Summary e Linked Issues
        content = (
            f"[Summary] {row['Summary']}. "
            f"[Linked Issues] {row['Linked Issues']}. "
            f"[Epic Link] {row['Epic Link']}."
        )

        # Divide o conteúdo em chunks menores (opcional)
        chunks = text_splitter.split_text(content)

        # Metadados básicos
        metadata = {
            "id": row["Key"],
            "status": row["Status"],
            "created": row["Created"],
            "reporter": row["Reporter"],
            "issue_type": row["Issue Type"],
        }

        # Adiciona colunas opcionais apenas se não estiverem vazias
        if pd.notna(row["Linked Issues"]):
            metadata["linked_issues"] = row["Linked Issues"]
        if pd.notna(row["Development"]):
            metadata["development"] = row["Development"]
        if pd.notna(row["Epic Link"]):
            metadata["epic_link"] = row["Epic Link"]
        if pd.notna(row["Epic Name"]):
            metadata["epic_name"] = row["Epic Name"]
        if pd.notna(row["Sprint"]):
            metadata["sprint"] = row["Sprint"]

        # Cria um documento para cada chunk (ou um único documento se não houver divisão)
        for chunk in chunks:
            document = Document(
                page_content=chunk,
                metadata=metadata
            )
            documents.append(document)

    print(f"📝 Total de documentos carregados: {len(documents)}")
    return documents

def split_documents(documents):
    """
    Divide documentos grandes em chunks menores.
    Neste caso, os documentos já correspondem a chunks individuais (linhas do CSV).
    """
    return documents  # Retorna os documentos diretamente

def add_to_chroma(chunks):
    """
    Adiciona ou atualiza chunks no banco de dados Chroma.
    """
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=get_embedding_function())

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
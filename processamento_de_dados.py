import pandas as pd
import json
from collections import defaultdict, Counter
from datetime import datetime

def carregar_csv(caminho_csv):
    """
    Carrega o arquivo CSV e retorna um DataFrame pandas.
    """
    try:
        return pd.read_csv(caminho_csv)
    except Exception as e:
        print(f"Erro ao carregar o CSV: {e}")
        return None

def contar_por_epic_link(df):
    """
    Conta o número de bugs por Epic Link.
    """
    if "Epic Link" in df.columns:
        return df['Epic Link'].value_counts().to_dict()
    return {}

def contar_bugs_por_release(df):
    """
    Conta o número total de bugs por release, baseado em padrões de texto na coluna Summary.
    Quando a release não está identificada, classifica como "release não identificada".
    """
    releases = defaultdict(int)
    
    if "Summary" in df.columns:
        for summary in df['Summary']:
            if isinstance(summary, str):
                # Extraindo release entre colchetes []. Ex.: [v1.6.0-sp35.3]
                if "[" in summary and "]" in summary:
                    release = summary.split(']')[0].split('[')[-1].strip()
                else:
                    release = "release não identificada"
                releases[release] += 1

    return dict(releases)

def contar_palavras_com_3_ou_mais_caracteres(df, coluna="Summary"):
    """
    Conta a frequência de palavras com 3 ou mais caracteres em uma coluna de texto.
    """
    if coluna in df.columns:
        palavras = ' '.join(df[coluna].dropna().astype(str)).split()
        palavras_filtradas = [palavra for palavra in palavras if len(palavra) >= 3]
        return Counter(palavras_filtradas)
    return {}

def categorizar_por_status(df):
    """
    Categoriza os bugs por status.
    """
    if "Status" in df.columns:
        return df['Status'].value_counts().to_dict()
    return {}

def categorizar_por_reporter(df):
    """
    Categoriza os bugs por reporter.
    """
    if "Reporter" in df.columns:
        return df['Reporter'].value_counts().to_dict()
    return {}

def calcular_tendencia_por_data(df, coluna_data="Created"):
    """
    Calcula a tendência de bugs criados por dia/mês.
    """
    if coluna_data in df.columns:
        df[coluna_data] = pd.to_datetime(df[coluna_data], errors='coerce')
        df = df.dropna(subset=[coluna_data])
        
        # Convertendo as chaves do dicionário para strings
        tendencia_dia = {str(data): contagem for data, contagem in df[coluna_data].dt.date.value_counts().sort_index().items()}
        tendencia_mes = {str(mes): contagem for mes, contagem in df[coluna_data].dt.to_period('M').value_counts().sort_index().items()}
        
        return {"por_dia": tendencia_dia, "por_mes": tendencia_mes}
    
    return {}

def preprocessar_dados(caminho_csv, exportar_json=False):
    """
    Realiza o pré-processamento dos dados e retorna o contexto em formato de dicionário.
    """
    df = carregar_csv(caminho_csv)
    if df is None:
        return None

    # Contagem por Epic Link.
    epic_link_counts = contar_por_epic_link(df)
    
    # Contagem de bugs por release (versão ajustada).
    bugs_por_release = contar_bugs_por_release(df)

    # Contagem de palavras com 3 ou mais caracteres.
    contagem_palavras = contar_palavras_com_3_ou_mais_caracteres(df)

    # Categorização por status.
    categorias_status = categorizar_por_status(df)

    # Categorização por reporter.
    categorias_reporter = categorizar_por_reporter(df)

    # Tendência de bugs por data.
    tendencia_data = calcular_tendencia_por_data(df)

    # Criação do contexto analítico.
    contexto_analitico = {
        "contagem_total_bugs": len(df),
        "contagem_epic_link": epic_link_counts,
        "contagem_bugs_release": bugs_por_release,
        "contagem_palavras": contagem_palavras,
        "categorias_status": categorias_status,
        "categorias_reporter": categorias_reporter,
        "tendencia_data": tendencia_data
    }

    if exportar_json:
        with open('contexto_analitico.json', 'w') as f:
            json.dump(contexto_analitico, f, indent=4)
        print("Contexto exportado para 'contexto_analitico.json'.")

    return contexto_analitico


if __name__ == "__main__":
    caminho_csv = "C:/Users/pedro/OneDrive/Documentos/basesdedefeitos/jirabugs.csv"
    contexto = preprocessar_dados(caminho_csv, exportar_json=True)
    print(json.dumps(contexto, indent=4))
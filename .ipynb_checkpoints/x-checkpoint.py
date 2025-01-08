# Importar bibliotecas
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Carregar os dados
df = pd.read_csv("C:/Users/pedro/OneDrive/Documentos/basesdedefeitos/jirabugs.csv")

# Verificar as primeiras linhas para entender o formato dos dados
df.head(10)
# Verificar as últimas linhas para entender o formato dos dados
df.tail(10)
# Definir os tipos de dados da coluna 'created' como datas
df['Created'] = pd.to_datetime(df['Created'], errors='coerce')
# Checar se existem linhas duplicadas
duplicate_rows = df.duplicated()

# Print the number of duplicate rows
print(f'Numero de linhas duplicadas: {duplicate_rows.sum()}')

# Remove duplicate rows
df_cleaned = df.drop_duplicates()

# Optionally, you can check the shape of the cleaned DataFrame
print(f'Formato do dataframe depois de remover duplicatas: {df_cleaned.shape}')
# Iterar cada coluna do dataframe e padronizar a formatação das strings passando para letras minusculas
for column in df.select_dtypes(include=['object']).columns:
    df[column] = df[column].str.strip().str.lower()
    # Definir celulas não preenchidas como 'missing' para maior organização
df.fillna('NULL')
# Verificar informações gerais (colunas, tipos de dados, valores nulos)
df.info()
# Verificar estatísticas descritivas (se houver dados numéricos)
df.describe()
# Contagem dos tipos de defeitos
defeitos_freq = df['Epic Link'].value_counts()
print(defeitos_freq)
# Top 10 defeitos mais comuns
defeitos_mais_comuns = defeitos_freq.head(10)
print("Defeitos mais comuns:\n", defeitos_mais_comuns)
# Top 10 defeitos menos comuns
defeitos_menos_comuns = defeitos_freq.tail(10)
print("Defeitos menos comuns:\n", defeitos_menos_comuns)
# Gráfico de frequência dos defeitos
plt.figure(figsize=(12, 6))
sns.barplot(x=defeitos_freq.index, y=defeitos_freq.values)
plt.xticks(rotation=90)
plt.xlabel("Epico")
plt.ylabel("Frequência")
plt.title("Frequência de Defeitos por Epico")
plt.show()
# Gráfico de distribuição de defeitos (Pareto)
plt.figure(figsize=(12, 6))
defeitos_freq.cumsum().plot(drawstyle="steps-post", color="blue", label="Cumulative Frequency")
defeitos_freq.plot(kind="bar", color="orange", alpha=0.7, label="Defect Count")
plt.ylabel("Frequência")
plt.title("Distribuição de Defeitos (Pareto)")
plt.legend()
plt.show()
# Resumir o número de defeitos por mês/ano
defeitos_por_mes = df.groupby(df['Created'].dt.to_period("M")).size()
defeitos_por_mes.plot(kind='line', marker='o', figsize=(12, 6))
plt.xlabel("Data")
plt.ylabel("Número de Defeitos")
plt.title("Frequência de Defeitos ao Longo do Tempo")
plt.show()
# Extrair a versão do sistema (release) da coluna 'Summary' usando regex
df['Release'] = df['Summary'].str.extract(r'\[(v[\d\.]+-[\w\d\.]+)\]')[0]
# Contar o número total de bugs por release
total_bug_counts = df.groupby('Release').size().reset_index(name='Total Bug Count')
# Gerar gráfico de bugs por release
plt.figure(figsize=(12, 6))
sns.barplot(data=total_bug_counts, x='Release', y='Total Bug Count', color='salmon')

# Ajustes de rótulos e título
plt.xticks(rotation=45, ha='right')
plt.xlabel("Versão (Release)")
plt.ylabel("Quantidade Total de Bugs")
plt.title("Quantidade Total de Bugs por Release")

# Mostrar o gráfico
plt.tight_layout()
plt.show()
# Contar o número de bugs para cada épico, agrupados por release
bug_counts = df.groupby(['Release', 'Epic Link']).size().reset_index(name='Bug Count')
# Gráfico de bugs por release e épico
plt.figure(figsize=(14, 8))
sns.barplot(data=bug_counts, x='Release', y='Bug Count', hue='Epic Link', dodge=True)

# Ajustes de rótulos e título
plt.xticks(rotation=45, ha='right')
plt.xlabel("Versão (Release)")
plt.ylabel("Quantidade de Bugs")
plt.title("Quantidade de Bugs por Release e Epico")

# Mostrar a legenda e o gráfico
plt.legend(title="Epic Link")
plt.tight_layout()
plt.show()

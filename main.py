import re
import pandas as pd
from datasets import load_dataset
from files.embeddings import gerar_embeddings 
from files.faiss_search import criar_faiss_index, buscar_no_faiss, buscar_por_produto
from files.preprocess import load_and_preprocess_data
from files.text_generation import gerar_resposta_por_produto
from sentence_transformers import SentenceTransformer
import numpy as np
from prettytable import PrettyTable

# Inicializa o modelo de embeddings
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# Modelo de geração de texto pré-treinado (GPT-2)
generator = pipeline('text-generation', model='gpt2')

# Função para limpar o texto das avaliações
def clean_text(text):
    text = re.sub(r'[^\w\s]', '', text)
    return text.lower()

def gerar_insights(df):
    limite_positivo = 4.0
    limite_negativo = 2.0

    # Converter colunas para string e preencher valores nulos
    df['product_name'] = df['product_name'].fillna('').astype(str)
    df['site_category_lv1'] = df['site_category_lv1'].fillna('').astype(str)

    # Agrupar por categoria e produto e calcular a média das avaliações
    grouped = df.groupby(['site_category_lv1', 'product_name']).agg(
        avg_rating=('overall_rating', 'mean'),
        review_count=('review_text', 'count')
    ).reset_index()

    insights = []
    total_positivos = 0
    total_negativos = 0

    for _, row in grouped.iterrows():
        categoria = row['site_category_lv1']
        produto = row['product_name']

        # Filtro para produto e categoria
        df_produto = df[(df['product_name'] == produto) & (df['site_category_lv1'] == categoria)]

        comentarios_positivos = df_produto[df_produto['overall_rating'] >= limite_positivo]
        comentarios_negativos = df_produto[df_produto['overall_rating'] <= limite_negativo]

        # Total de avaliações positivas e negativas
        total_positivos += len(comentarios_positivos)
        total_negativos += len(comentarios_negativos)

        # Criar tabelas para comentários positivos e negativos
        tabela_positivos = PrettyTable()
        tabela_positivos.field_names = ["Comentário Positivo", "Nota"]
        for _, comentario in comentarios_positivos.iterrows():
            tabela_positivos.add_row([comentario['review_text'], comentario['overall_rating']])

        tabela_negativos = PrettyTable()
        tabela_negativos.field_names = ["Comentário Negativo", "Nota"]
        for _, comentario in comentarios_negativos.iterrows():
            tabela_negativos.add_row([comentario['review_text'], comentario['overall_rating']])

        insights.append({
            'Categoria': categoria,
            'Produto': produto,
            'Média de Avaliação': round(row['avg_rating'], 2),
            'Número de Avaliações': row['review_count'],
            'Total Avaliações Positivas': len(comentarios_positivos),
            'Total Avaliações Negativas': len(comentarios_negativos),
            'Tabela Positivos': tabela_positivos,
            'Tabela Negativos': tabela_negativos
        })

    # Exibir as tabelas geradas
    for insight in insights:
        print(f"\nCategoria: {insight['Categoria']}, Produto: {insight['Produto']}, "
              f"Média de Avaliação: {insight['Média de Avaliação']}, "
              f"Número de Avaliações: {insight['Número de Avaliações']}")
        
        print(f"Total de Avaliações Positivas: {insight['Total Avaliações Positivas']}")
        print(f"Total de Avaliações Negativas: {insight['Total Avaliações Negativas']}")

        print("\nComentários Positivos:")
        print(insight['Tabela Positivos'])

        print("\nComentários Negativos:")
        print(insight['Tabela Negativos'])

    return insights, total_positivos, total_negativos

def main():
    # Carregar o dataset de avaliações de ambos os arquivos CSV
    df_chat_data = load_and_preprocess_data(r'data/chat_data.csv')
    df_chat_data_processado = load_and_preprocess_data(r'data/chat_data_processado.csv')

    # Concatenar os dois DataFrames
    df_combined = pd.concat([df_chat_data, df_chat_data_processado], ignore_index=True)

    df_combined['review_text'] = df_combined['review_text'].fillna('')
    df_combined['review_text'] = df_combined['review_text'].apply(clean_text)
    df_combined = df_combined[df_combined['overall_rating'].notnull() & df_combined['product_name'].notnull()]

    # Gerar insights de todos os produtos
    insights, total_positivos, total_negativos = gerar_insights(df_combined)

    # Calcular a média geral de avaliações
    total_avaliacoes = total_positivos + total_negativos
    print(f"\nTotal Geral de Avaliações Positivas: {total_positivos}")
    print(f"Total Geral de Avaliações Negativas: {total_negativos}")
    print(f"Total Geral de Avaliações: {total_avaliacoes}")

    # Carregar modelo e gerar embeddings
    model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
    text_columns = ['product_name', 'review_text', 'site_category_lv1']
    embeddings_dict = gerar_embeddings(df_combined, text_columns)
    index = criar_faiss_index(np.array(embeddings_dict['review_text']))

    print("Bem-vindo ao Chat ATM! Digite 'sair' para encerrar.")

    while True:
        consulta_produto = input("Qual produto você deseja receber a avaliação? ").strip()

        if consulta_produto.lower() == 'sair':
            print("Encerrando o Chat ATM. Até mais!")
            break

        # Ajustar a busca por produto
        resultados_chat = buscar_por_produto(df_combined, index, consulta_produto, model)

        if resultados_chat:
            resposta_gerada = gerar_resposta_por_produto(resultados_chat)
            print(f"Resposta gerada: {resposta_gerada}")
        else:
            print("Produto não encontrado ou sem avaliações.")

        # Arredondar o tempo de busca para o chat
        tempo_arredondado_chat = round(tempo_busca_chat, 2)
        print(f"Tempo de busca para o chat: {tempo_arredondado_chat:.2f} segundos")

if __name__ == "__main__":
    main()

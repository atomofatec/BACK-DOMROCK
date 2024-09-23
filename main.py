import re
import pandas as pd
from datasets import load_dataset
from files.embeddings import gerar_embeddings
from files.faiss_search import criar_faiss_index, buscar_por_produto
from files.preprocess import load_and_preprocess_data
from files.text_generation import gerar_resposta_por_produto
from sentence_transformers import SentenceTransformer
import numpy as np

def clean_text(text):
    text = re.sub(r'[^\w\s]', '', text)  
    return text.lower()

def main():
    # Carregar o dataset e pré-processar os dados
    dataset = load_dataset('ruanchaves/b2w-reviews01')
    df = pd.DataFrame(dataset['train'])

    # Aplicar a função de limpeza nas avaliações
    df['review_text'] = df['review_text'].fillna('') 
    df['review_text'] = df['review_text'].apply(clean_text)

    # Verificar se há valores nulos nas colunas relevantes
    df = df[df['overall_rating'].notnull() & df['product_name'].notnull()]

    # Agrupar por 'site_category_lv1' e 'product_name'
    grouped = df.groupby(['site_category_lv1', 'product_name']).agg(
        avg_rating=('overall_rating', 'mean'), 
        review_count=('review_text', 'count')  
    ).reset_index()

    # Classificar produtos por categoria e avaliação média
    grouped = grouped.sort_values(['site_category_lv1', 'avg_rating'], ascending=[True, False])
    top_products_per_category = grouped.groupby('site_category_lv1').head(1)

    # Exibir as categorias e seus produtos mais bem avaliados
    print(top_products_per_category[['site_category_lv1', 'product_name', 'avg_rating', 'review_count']])

    # Adicionar os produtos mais bem avaliados ao dataframe principal
    df = df.merge(top_products_per_category[['site_category_lv1', 'product_name']], 
                  on=['site_category_lv1', 'product_name'], 
                  how='left', 
                  indicator=True)
    
    # Manter apenas os produtos que estão entre os mais bem avaliados
    df = df[df['_merge'] == 'both'].drop(columns=['_merge'])

    # Inicializa o modelo de embeddings
    model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

    # Gerar embeddings para as colunas relevantes incluindo os produtos mais bem avaliados
    text_columns = ['product_name', 'review_text', 'site_category_lv1']
    embeddings_dict = gerar_embeddings(df, text_columns)

    # Criar índice FAISS para os embeddings de review_text
    index = criar_faiss_index(np.array(embeddings_dict['review_text']))

    # Carregar e pré-processar os dados do chat_data.csv
    df_chat_data = load_and_preprocess_data(r'data\chat_data.csv')

    # Definir as colunas de texto para geração de embeddings
    text_columns_chat = ['product_name_processado', 'product_brand_processado', 
                         'site_category_lv1_processado', 'site_category_lv2_processado', 
                         'review_title_processado', 'recommend_to_a_friend_processado', 
                         'review_text_processado', 'reviewer_gender_processado', 
                         'reviewer_state_processado']

    # Gerar embeddings para os dados do chat
    embeddings_dict_chat = gerar_embeddings(df_chat_data, text_columns_chat)

    # Criar índice FAISS para os embeddings de review_text_processado
    index_chat = criar_faiss_index(embeddings_dict_chat['review_text_processado'])

    print("Bem-vindo ao Chat ATM! Digite 'sair' para encerrar.")

    while True:
        consulta_produto = input("Qual produto você deseja receber a avaliação? ")

        if consulta_produto.lower() == 'sair':
            print("Encerrando o Chat ATM. Até mais!")
            break

        # Consultar por produto
        resultados_chat = buscar_por_produto(df_chat_data, index_chat, consulta_produto, model)

        # Gerar resposta com base nas avaliações encontradas
        if resultados_chat:
            resposta_gerada = gerar_resposta_por_produto(resultados_chat)
            print(f"Resposta gerada: {resposta_gerada}")
        else:
            print("Produto não encontrado ou sem avaliações.")

if __name__ == "__main__":
    main()
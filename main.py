import os
import re
import numpy as np
import pandas as pd
from datasets import load_dataset
from files.preprocess import load_and_preprocess_data
from files.embeddings import gerar_embeddings
from files.faiss_search import criar_faiss_index, buscar_por_produto
from files.text_generation import gerar_resposta_por_produto
from sentence_transformers import SentenceTransformer
import faiss  # Biblioteca FAISS para lidar com os índices de vetor

# Função para limpar texto
def clean_text(text):
    text = re.sub(r'[^\w\s]', '', text)
    return text.lower()

# Função para salvar o índice FAISS no disco
def salvar_faiss_index(index, file_path):
    faiss.write_index(index, file_path)

# Função para carregar o índice FAISS do disco
def carregar_faiss_index(file_path):
    if os.path.exists(file_path):
        return faiss.read_index(file_path)
    return None

# Função para salvar embeddings em arquivo
def salvar_embeddings(embeddings_dict, file_path):
    np.save(file_path, embeddings_dict)

# Função para carregar embeddings de arquivo
def carregar_embeddings(file_path):
    if os.path.exists(file_path):
        return np.load(file_path, allow_pickle=True).item()
    return None

def main():
    # Caminhos para salvar embeddings e índice FAISS
    faiss_index_path = 'data/faiss_index.index'
    embeddings_path = 'data/embeddings.npy'

    # Tentar carregar o índice FAISS salvo
    index = carregar_faiss_index(faiss_index_path)

    # Tentar carregar os embeddings salvos
    embeddings_dict = carregar_embeddings(embeddings_path)

    if index is None or embeddings_dict is None:
        # Carregar e pré-processar os dados do dataset
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

        # Gerar embeddings para as colunas relevantes
        text_columns = ['product_name', 'review_text', 'site_category_lv1']
        embeddings_dict = gerar_embeddings(df, text_columns)

        # Criar índice FAISS para os embeddings de review_text
        index = criar_faiss_index(np.array(embeddings_dict['review_text']))

        # Carregar e pré-processar os dados do chat_data.csv
        df_chat_data = load_and_preprocess_data(r'data/chat_data.csv')

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

        # Salvar os embeddings e o índice FAISS no disco
        salvar_embeddings(embeddings_dict, embeddings_path)
        salvar_faiss_index(index, faiss_index_path)

    else:
        print("Índice FAISS e embeddings carregados com sucesso.")

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

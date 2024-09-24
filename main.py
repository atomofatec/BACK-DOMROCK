import os
import numpy as np
from files.preprocess import load_and_preprocess_data
from files.embeddings import gerar_embeddings
from files.faiss_search import criar_faiss_index, buscar_por_produto
from files.text_generation import gerar_resposta_por_produto
from sentence_transformers import SentenceTransformer
import faiss  # Biblioteca FAISS para lidar com os índices de vetor

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

# Caminhos para salvar embeddings e índice FAISS
faiss_index_path = 'data/faiss_index.index'
embeddings_path = 'data/embeddings.npy'

# Tentar carregar o índice FAISS salvo
index = carregar_faiss_index(faiss_index_path)

# Tentar carregar os embeddings salvos
embeddings_dict = carregar_embeddings(embeddings_path)

if index is None or embeddings_dict is None:
    # Carregar e pré-processar os dados
    df = load_and_preprocess_data(r'data/chat_data.csv')

    model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

    # Definir as colunas de texto para geração de embeddings
    text_columns = ['product_name_processado', 'product_brand_processado', 'site_category_lv1_processado', 
                    'site_category_lv2_processado', 'review_title_processado', 'recommend_to_a_friend_processado', 
                    'review_text_processado', 'reviewer_gender_processado', 'reviewer_state_processado']

    # Gerar embeddings se eles não existirem
    if embeddings_dict is None:
        embeddings_dict = gerar_embeddings(df, text_columns)

        # Salvar os embeddings no disco
        salvar_embeddings(embeddings_dict, embeddings_path)

    # Criar o índice FAISS para os embeddings
    index = criar_faiss_index(embeddings_dict['review_text_processado'])

    # Salvar o índice FAISS no disco
    salvar_faiss_index(index, faiss_index_path)
else:
    print("Índice FAISS e embeddings carregados com sucesso.")

# Consulta de exemplo
consulta_produto = "Copo Acrílico Com Canudo 500ml Rocie"
resultados = buscar_por_produto(df, index, consulta_produto, model, k=10)

# Gerar resposta com GPT-2
if resultados:
    resposta_gerada = gerar_resposta_por_produto(resultados)
    print(f"Resposta gerada: {resposta_gerada}")
else:
    print("Produto não encontrado ou sem avaliações.")

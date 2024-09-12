from preprocess import load_and_preprocess_data
import embeddings
from faiss_search import criar_faiss_index
from faiss_search import buscar_por_produto
from text_generation import gerar_resposta_por_produto

# Carregar e pré-processar os dados
df = load_and_preprocess_data('chat_data.csv')

# Definir as colunas de texto para geração de embeddings
text_columns = ['product_name_processado', 'product_brand_processado', 'site_category_lv1_processado', 
                'site_category_lv2_processado', 'review_title_processado', 'recommend_to_a_friend_processado', 
                'review_text_processado', 'reviewer_gender_processado', 'reviewer_state_processado']

# Gerar embeddings
embeddings_dict = embeddings.gerar_embeddings(df, text_columns)

# Criar o índice Faiss para os embeddings da coluna review_text_processado
index = criar_faiss_index(embeddings_dict['review_text_processado'])

# Consultar por todas as avaliações de um produto específico
consulta_produto = "Copo Acrílico Com Canudo 500ml Rocie"
resultados = buscar_por_produto(df, index, consulta_produto, embeddings.model, k=10)

# Gerar a resposta agregada com base nas notas encontradas
if resultados:
    resposta_gerada = gerar_resposta_por_produto(resultados)
    print(f"Resposta gerada: {resposta_gerada}")
else:
    print("Produto não encontrado ou sem avaliações.")

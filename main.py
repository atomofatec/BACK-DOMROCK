from files.preprocess import load_and_preprocess_data
from files.embeddings import gerar_embeddings
from files.faiss_search import criar_faiss_index
from files.faiss_search import buscar_por_produto
from files.text_generation import gerar_resposta_por_produto
from sentence_transformers import SentenceTransformer

# criar ambiente virtual em python (opcional)
# instalar os pacotes com o comando abaixo (obrigatório):
# pip install -r requirements.txt

# carregar e pré-processar os dados
df = load_and_preprocess_data(r'data\chat_data.csv')

model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# Definir as colunas de texto para geração de embeddings
text_columns = ['product_name_processado', 'product_brand_processado', 'site_category_lv1_processado', 
                'site_category_lv2_processado', 'review_title_processado', 'recommend_to_a_friend_processado', 
                'review_text_processado', 'reviewer_gender_processado', 'reviewer_state_processado']

# gerar embeddings
embeddings_dict = gerar_embeddings(df, text_columns)

# criar índice faiss para os embeddings
index = criar_faiss_index(embeddings_dict['review_text_processado'])

# consultar por comentário similar (o comentário está sendo passado diretamente no código, criar interface no prompt para interação)
consulta_produto = "Copo Acrílico Com Canudo 500ml Rocie"
resultados = buscar_por_produto(df, index, consulta_produto, model, k=10)

# gerar resposta com gpt2
if resultados:
    resposta_gerada = gerar_resposta_por_produto(resultados)
    print(f"Resposta gerada: {resposta_gerada}")
else:
    print("Produto não encontrado ou sem avaliações.")

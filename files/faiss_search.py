import faiss
import numpy as np
from .preprocess import preprocess_text

# define a dimensão dos vetores, cria um índice faiss com uma distância de busca definida e adiciona os embeddings processados ao índice
def criar_faiss_index(embeddings):
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings))
    return index

# pré-processa e gera um embedding para a consulta para comparar com os embeddings já salvos e buscar por valores semelhantes. adiciona o resultado no final da lista
# possivelmente vai precisar de mudanças para retornos mais precisos
def buscar_por_produto(df, index, produto, model, k=10):
    produto_processado = preprocess_text(produto)  # Pré-processa o nome do produto
    produto_embedding = model.encode([produto_processado])  # Gera embedding
    distances, indices = index.search(np.array(produto_embedding), k=k)  # Busca no índice
    
    resultados = []
    for i in range(len(indices[0])):  # Coleta todas as avaliações correspondentes
        resultado = df.iloc[indices[0][i]]
        if resultado['product_name'] == produto:  # Verifica se o nome do produto corresponde
            resultados.append({
                'produto': resultado['product_name'],
                'nota': resultado['overall_rating'],
                'comentário': resultado['review_text'],
                'data_submissão': resultado['submission_date'],
                'título_revisão': resultado['review_title'],
                'recomenda_para_amigo': resultado['recommend_to_a_friend'],

                'site_category_lv1': resultado['site_category_lv1'],
                'site_category_lv2': resultado['site_category_lv2'],

            })
    return resultados

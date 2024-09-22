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
def buscar_por_produto(df, index, consulta, model):
    consulta_processada = preprocess_text(consulta)  # Pré-processa a consulta
    consulta_embedding = model.encode([consulta_processada])  # Gera embedding
    distances, indices = index.search(np.array(consulta_embedding), k=10)  # Busca no índice
    
    # Coleta o índice do produto mais similar
    idx = indices[0][0]
    resultados = df.iloc[idx]
    
    return {
        'produto': resultados['product_name'],
        'nota': resultados['overall_rating'],
        'comentário': resultados['review_text']
    }
    
    return resultados
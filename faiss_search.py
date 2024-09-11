import faiss
import numpy as np
from preprocess import preprocess_text

# define a dimensão dos vetores, cria um índice faiss com uma distância de busca definida e adiciona os embeddings processados ao índice
def criar_faiss_index(embeddings):
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings))
    return index

# pré-processa e gera um embedding para a consulta para comparar com os embeddings já salvos e buscar por valores semelhantes. adiciona o resultado no final da lista
# possivelmente vai precisar de mudanças para retornos mais precisos
def buscar_no_faiss(df, index, consulta, model, k=1):
    consulta_processada = preprocess_text(consulta)
    consulta_embedding = model.encode([consulta_processada])
    distances, indices = index.search(np.array(consulta_embedding), k=k)
    
    resultados = []
    for i in range(k):
        resultado = df.iloc[indices[0][i]]
        resultados.append({
            'produto': resultado['product_name'],
            'comentário': resultado['review_text'],
            'nota': resultado['overall_rating']
        })
    return resultados
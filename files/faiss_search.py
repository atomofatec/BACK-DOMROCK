import faiss
import numpy as np
from .preprocess import preprocess_text

def criar_faiss_index(embeddings):
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings))
    return index

def buscar_por_produto(df, index, consulta, model, k=10):
    consulta_processada = preprocess_text(consulta)  # Pré-processa a consulta
    consulta_embedding = model.encode([consulta_processada])  # Gera embedding
    distances, indices = index.search(np.array(consulta_embedding), k=k)  # Busca no índice

    resultados = []
    for idx in indices[0]:  # Coleta todas as avaliações correspondentes
        resultado = df.iloc[idx]
        resultados.append({
            'produto': resultado['product_name'],
            'nota': resultado['overall_rating'],
            'comentário': resultado['review_text']
        })
    return resultados
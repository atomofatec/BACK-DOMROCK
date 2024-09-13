import faiss
import numpy as np
from preprocess import preprocess_text  # Presumo que você tenha esse arquivo

# Define a dimensão dos vetores, cria um índice FAISS e adiciona os embeddings processados ao índice
def criar_faiss_index(embeddings):
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings))
    return index

# Função para buscar no FAISS
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

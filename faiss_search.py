import faiss
import numpy as np
from preprocess import preprocess_text

# Função que cria o índice Faiss para os embeddings
def criar_faiss_index(embeddings):
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings))
    return index

# Função que busca avaliações similares por nome do produto
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
                'comentário': resultado['review_text']
            })
    return resultados

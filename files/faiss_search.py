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
            'comentário': resultado['review_text'],
            'data_submissão': resultado.get('submission_date', 'Data não disponível'),
            'título_revisão': resultado.get('review_title', 'Sem título'),
            'recomenda_para_amigo': resultado.get('recommend_to_a_friend', 'Sem recomendação'),
            'site_category_lv1': resultado.get('site_category_lv1', 'Categoria 1 não disponível'),
            'site_category_lv2': resultado.get('site_category_lv2', 'Categoria 2 não disponível')
        })
    
    return resultados

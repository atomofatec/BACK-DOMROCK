import faiss
import numpy as np
from .preprocess import preprocess_text
import time

def arredondar_tempo(tempo):
    return round(tempo, 4)

def criar_faiss_index(embeddings):
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings))
    return index

def buscar_no_faiss(df, index, consulta, model, k=10):
    tempo_inicio = time.time()  # Início da contagem do tempo
    consulta_processada = preprocess_text(consulta)  # Pré-processa o texto da consulta
    consulta_embedding = model.encode([consulta_processada])  # Gera embedding
    distances, indices = index.search(np.array(consulta_embedding), k=k)  # Busca no índice

    resultados = []
    for i in range(len(indices[0])):  # Coleta todas as avaliações correspondentes
        resultado = df.iloc[indices[0][i]]
        resultados.append({
            'produto': resultado['product_name'],
            'nota': resultado['overall_rating'],
            'comentário': resultado['review_text']
        })
    
    tempo_busca = time.time() - tempo_inicio  # Tempo total de busca
    tempo_busca_arredondado = arredondar_tempo(tempo_busca)
    
    print(f"Tempo de busca: {tempo_busca_arredondado} segundos")
    if tempo_busca_arredondado > 2:
        print("Erro: Tempo de busca excedeu 2 segundos.")
        return []  # Retorna uma lista vazia se exceder o tempo
    return resultados

def buscar_por_produto(df, index, produto, model, k=10):
    tempo_inicio = time.time()  # Início da contagem do tempo
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

    tempo_busca = time.time() - tempo_inicio  # Tempo total de busca
    tempo_busca_arredondado = arredondar_tempo(tempo_busca)
    
    print(f"Tempo de busca para o produto: {tempo_busca_arredondado} segundos")
    if tempo_busca_arredondado > 2:
        print("Erro: Tempo de busca para o produto excedeu 2 segundos.")
        return []  # Retorna uma lista vazia se exceder o tempo
    return resultados

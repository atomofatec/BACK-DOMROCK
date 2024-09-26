from sentence_transformers import SentenceTransformer
import pandas as pd

# Modelo pré-treinado de embeddings de sentenças
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# Função para gerar e armazenar embeddings para cada coluna de texto
def gerar_embeddings(df, text_columns):
    embeddings_dict = {}
    
    for col in text_columns:
        if col in df.columns:
            df[col] = df[col].fillna('').astype(str)  # Garantir que os dados são strings
            embeddings = model.encode(df[col].tolist(), show_progress_bar=True)  # Adiciona uma barra de progresso
            embeddings_dict[col] = embeddings
    
    return embeddings_dict

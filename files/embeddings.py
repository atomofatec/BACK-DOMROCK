from sentence_transformers import SentenceTransformer

# modelo pré-treinado de embeddings de sentenças
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# função para gerar e armazenar embeddings para cada coluna de texto
def gerar_embeddings(df, text_columns):
    embeddings_dict = {}
    
    for col in text_columns:
        if col in df.columns:
            df[col] = df[col].fillna('')
            embeddings = model.encode(df[col].tolist())
            embeddings_dict[col] = embeddings
    
    return embeddings_dict
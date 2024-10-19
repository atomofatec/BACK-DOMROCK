import pandas as pd
from tqdm import tqdm
import torch
from chromadb import Client
import chromadb
import uuid  # Para gerar IDs únicos
import numpy as np  # Para normalização
import ast  # Para conversão segura de strings em listas

# Função para ler o CSV com os chunks
def read_chunks_csv(file_path):
    df = pd.read_csv(file_path)
    # Converte as strings que representam listas em listas reais
    chunks = df['valid_chunks'].dropna().tolist()
    flat_chunks = []
    for sublist in chunks:
        try:
            flat_chunks.extend(ast.literal_eval(sublist))
        except (ValueError, SyntaxError):
            print(f"Erro ao avaliar a string: {sublist}")
    return flat_chunks  

# Função para normalizar os embeddings
def normalize_embeddings(embeddings):
    return embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

# Função para gerar embeddings
def generate_embeddings(chunks, batch_size=64):
    from transformers import AutoTokenizer, AutoModel  
    model = AutoModel.from_pretrained('neuralmind/bert-large-portuguese-cased')  
    tokenizer = AutoTokenizer.from_pretrained('neuralmind/bert-large-portuguese-cased', do_lower_case=False)  
    model.eval()
    embeddings = []  
    for i in tqdm(range(0, len(chunks), batch_size), desc="Gerando embeddings"):
        batch = chunks[i:i + batch_size]
        inputs = tokenizer(batch, padding=True, truncation=True, return_tensors='pt')  
        with torch.no_grad():  
            outputs = model(**inputs)
        batch_embeddings = outputs[0].mean(dim=1)  
        embeddings.extend(batch_embeddings.numpy())  
    
    # Normaliza os embeddings
    normalized_embeddings = normalize_embeddings(np.array(embeddings))
    return normalized_embeddings  

# Função para salvar embeddings no ChromaDB
def save_embeddings_to_chromadb(embeddings, chunks):
    client = chromadb.PersistentClient(path="data/chromadb")
    collection_name = "product_reviews"

    # Verifica se a coleção já existe e tenta excluí-la
    try:
        client.get_collection(name=collection_name)
        client.delete_collection(name=collection_name)  # Exclui a coleção existente
        print(f"Coleção {collection_name} excluída com sucesso.")
    except Exception as e:
        print(f"A coleção {collection_name} não existia ou foi excluída com sucesso.")

    # Cria uma nova coleção
    collection = client.create_collection(name=collection_name)

    # Adiciona os embeddings e os chunks à nova coleção
    for chunk, embedding in zip(chunks, embeddings):
        embedding_id = str(uuid.uuid4())  # Gera um ID único para cada documento
        
        collection.add(
            documents=[chunk],
            embeddings=[embedding],
            metadatas=[{"text": chunk}],
            ids=[embedding_id]
        )
        print(f"Embedding com ID {embedding_id} adicionado.")

    print(f"{len(embeddings)} embeddings salvos no ChromaDB.")

def verify_embeddings_in_chromadb(collection_name="product_reviews"):
    client = chromadb.PersistentClient(path="data/chromadb")

    # Tenta obter a coleção
    collection = client.get_collection(name=collection_name)

    # Conta o número de documentos na coleção
    num_documents = len(collection.get()['documents'])
    print(f"Número de documentos na coleção '{collection_name}': {num_documents}")

    # Recupera e exibe alguns documentos
    if num_documents > 0:
        documents = collection.get()['documents'][:5]  # Obtém os primeiros 5 documentos
        print("Primeiros 5 documentos na coleção:")
        for doc in documents:
            print(doc)

# Caminho para o arquivo CSV que contém os chunks
chunks_csv_path = r'data\data_with_chunks.csv'  

# Chama a função para ler os chunks do CSV
chunks = read_chunks_csv(chunks_csv_path)  

# Gera os embeddings a partir dos chunks lidos
embeddings = generate_embeddings(chunks)  

# Salva os embeddings no ChromaDB
save_embeddings_to_chromadb(embeddings, chunks)  

# Cria um DataFrame para armazenar os embeddings e os chunks originais
embeddings_df = pd.DataFrame({
    'chunk': chunks,  
    'embedding': [emb.tolist() for emb in embeddings]  
})

# Salva os embeddings em um arquivo CSV
embeddings_csv_path = r'data\embeddings.csv'  
embeddings_df.to_csv(embeddings_csv_path, index=False)  

# Verifica se os embeddings foram salvos
verify_embeddings_in_chromadb()

# Exibe os resultados
print(f"Número total de chunks: {len(chunks)}")  
print(f"Número total de embeddings gerados: {len(embeddings)}")  
print(f"Embeddings salvos em: {embeddings_csv_path}")  

# Exibe os 5 primeiros chunks e seus embeddings
for i, row in embeddings_df.head(5).iterrows():  
    print(f"Chunk {i+1}:\n{row['chunk']}\n")  
    print(f"Embedding {i+1}: {row['embedding']}\n")  
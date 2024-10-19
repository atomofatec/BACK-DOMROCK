import torch
from transformers import AutoTokenizer, AutoModel
import chromadb

# Função para gerar embeddings
def generate_embedding(text):
    model = AutoModel.from_pretrained('neuralmind/bert-large-portuguese-cased')
    tokenizer = AutoTokenizer.from_pretrained('neuralmind/bert-large-portuguese-cased', do_lower_case=False)
    model.eval()

    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)

    embedding = outputs[0].mean(dim=1).squeeze().tolist()  # Altere para mean(dim=1)
    print(f"Embedding gerado para '{text}': {embedding}")  # Imprimir embedding
    return embedding

# Função para buscar documentos similares
def search_similar_documents(query, collection_name="product_reviews", top_k=5):
    client = chromadb.PersistentClient(path="data/chromadb")
    collection = client.get_collection(name=collection_name)

    # Gera o embedding da consulta
    query_embedding = generate_embedding(query)

    # Realiza a busca por similaridade
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k
    )

    # Cálculo de similaridade de cosseno
    for i, (doc, score, doc_id) in enumerate(zip(results['documents'], results['distances'], results['ids'])):
        print(f"{i+1}. ID: {doc_id}, Documento: {doc}, Similaridade (Distância Coseno): {score}")

# Exemplo de uso
if __name__ == "__main__":
    search_query = "basico  gostar menos  aparencia propaganda"
    search_similar_documents(search_query)
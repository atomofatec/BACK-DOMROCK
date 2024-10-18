# chunking.py
import pandas as pd

# Função para criar chunks
def create_chunks(text, chunk_size=100, overlap_size=10):
    # Divide o texto em palavras
    words = text.split()
    chunks = []
    
    # Cria os chunks com sobreposição
    for i in range(0, len(words), chunk_size - overlap_size):
        chunk = ' '.join(words[i:i + chunk_size])
        if chunk:  # Verifica se o chunk não está vazio
            chunks.append(chunk)
    return chunks

# Função para validar chunks
def validate_chunks(chunks, max_chunk_size=100):
    valid_chunks = []
    for chunk in chunks:
        if len(chunk.split()) <= max_chunk_size and len(chunk.strip()) > 0:
            valid_chunks.append(chunk)
        else:
            print(f"Chunk inválido: '{chunk}' (tamanho: {len(chunk.split())} palavras)")
    return valid_chunks

# Função principal
def main():
    # Carregar o DataFrame pré-processado
    df = pd.read_csv(r'data\chat_data_processado.csv')

    # Aplicar a função de chunking na coluna review_text
    df['chunks'] = df['review_text'].apply(lambda x: create_chunks(x, chunk_size=100, overlap_size=10))

    # Validar os chunks e contar os resultados
    df['valid_chunks'] = df['chunks'].apply(validate_chunks)
    
    # Relatório de resultados
    total_chunks = df['valid_chunks'].apply(len).sum()
    print(f"Número total de chunks válidos: {total_chunks}")

    # Calcular e exibir a média de palavras por chunk
    total_words = sum(len(chunk.split()) for valid_list in df['valid_chunks'] for chunk in valid_list)
    average_words = total_words / total_chunks if total_chunks > 0 else 0
    print(f"Média de palavras por chunk: {average_words:.2f}")

    # Salvar o DataFrame com chunks em um novo arquivo .csv
    df.to_csv(r'data\data_with_chunks.csv', index=False)
    print("Arquivo com chunks salvo como 'data_with_chunks.csv'.")

if __name__ == "__main__":
    main()

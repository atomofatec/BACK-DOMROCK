# chunking.py
import pandas as pd
from langchain.text_splitter import RecursiveCharacterTextSplitter
from nltk.corpus import stopwords

# função para criar chunks usando Langchain
# define o tamanho da chunk com base no tamanho médio dos comentários
# comentários pequenos = chunks menores para aumentar eficiência
# também define o tamanho de overlapping
# a cada 50 palavras, 10 irão se repetir para manter o contexto das reviews
# aumentar conforme necessário para aprimorar eficiência
def create_chunks_with_langchain(text, chunk_size=50, overlap_size=10):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap_size)
    chunks = text_splitter.split_text(text)
    return chunks

# função para validar chunks
# chunks com menos de 5 palavras são descartadas pois não têm conteúdo o suficiente
# chunks com mais de 50 palavras serão quebradas
def validate_chunks(chunks, min_chunk_size=5, max_chunk_size=50):
    valid_chunks = []
    for chunk in chunks:
        word_count = len(chunk.split())

        # valida se o chunk está dentro do tamanho permitido
        if word_count < min_chunk_size:
            print(f"Chunk descartado por ser muito pequeno: '{chunk}' ({word_count} palavras)")
            continue
        if word_count > max_chunk_size:
            print(f"Chunk muito grande: '{chunk}' ({word_count} palavras), considere ajustar chunk_size ou overlap_size.")
        
        # verifica se o chunk é composto apenas de stopwords
        if all(word in stopwords.words('portuguese') for word in chunk.split()):
            print(f"Chunk descartado por conter apenas stopwords: '{chunk}'")
            continue
        
        # remover chunks duplicados
        if chunk not in valid_chunks:
            valid_chunks.append(chunk)
        else:
            print(f"Chunk duplicado descartado: '{chunk}'")
    
    return valid_chunks

# função principal
def main():
    # carregar o dataframe pré-processado
    df = pd.read_csv(r'data\chat_data_processado.csv', encoding='utf-8')

    # aplicar a função de chunking na coluna review_text usando langchain
    df['chunks'] = df['review_text'].apply(lambda x: create_chunks_with_langchain(x, chunk_size=50, overlap_size=10))

    # validar os chunks e contar os resultados
    df['valid_chunks'] = df['chunks'].apply(validate_chunks)
    
    # relatório de resultados
    total_chunks = df['valid_chunks'].apply(len).sum()
    print(f"Número total de chunks válidos: {total_chunks}")

    # calcular e exibir a média de palavras por chunk
    total_words = sum(len(chunk.split()) for valid_list in df['valid_chunks'] for chunk in valid_list)
    average_words = total_words / total_chunks if total_chunks > 0 else 0
    print(f"Média de palavras por chunk: {average_words:.2f}")

    # salvar o dataframe com chunks em um novo arquivo .csv
    df.to_csv(r'data\data_with_chunks.csv', index=False)
    print("Arquivo com chunks salvo como 'data_with_chunks.csv'.")

if __name__ == "__main__":
    main()
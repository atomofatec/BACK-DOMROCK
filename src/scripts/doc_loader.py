import os
import pandas as pd
from langchain_community.document_loaders import CSVLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from .preprocess import normalize_text, remove_stopwords

# define as colunas do csv que serão usadas
useful_data = ['product_name', 'product_brand', 'site_category_lv1', 'site_category_lv2', 'overall_rating', 'review_text']

# carrega o csv
def load_data(file_path):
    # lê o arquivo csv
    df = pd.read_csv(file_path)
    
    # exclui as colunas que não serão usadas
    df_reduced = df.drop(columns=[col for col in df.columns if col not in useful_data])

    # limpa e normaliza o texto
    for column in useful_data:
        df_reduced[column] = df_reduced[column].apply(lambda x: normalize_text(str(x)))
        df_reduced[column] = df_reduced[column].apply(lambda x: remove_stopwords(str(x)))

    # define o nome e o caminho do arquivo de saída
    result_file_name = 'data_processed.csv'  # nome do arquivo de saída
    new_file_path = os.path.join('data', result_file_name)  # caminho para salvar o arquivo de saída na pasta 'data'
    
    # salva os dados em csv
    df_reduced.to_csv(new_file_path, index=False)

    return new_file_path

def load_and_chunk(file_path):
    # gera os documentos com o csv processado
    loader = CSVLoader(file_path=file_path, encoding='utf-8', csv_args={
        'delimiter': ',', 'quotechar': '"', 'fieldnames': useful_data
    }) # classe do langchain para manipular os dados
    # delimiter define o que separa as colunas do csv (no caso, uma vírgula)
    # quotechar define o caractere que envolve strings, permitindo que uma string tenha vírgulas sem ser divididas em colunas
    # fieldnames são os nomes das colunas que serão carregadas

    docs = loader.load() # retorna os documentos gerados

    # chunkeniza e adiciona overlap nos dados
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)
    splits = text_splitter.split_documents(docs)

    return splits
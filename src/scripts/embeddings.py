from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

def generate_embeddings(splits):
    # inicia a classe do langchain para geração dos embeddings
    hf = HuggingFaceEmbeddings(
        model_name='sentence-transformers/all-MiniLM-L6-v2',  # modelo da sentence transformers voltado para pt-br para a geração de embeddings
        model_kwargs={'device': 'cpu'},  # define o dispositivo que está sendo usado (CPU, GPU etc)
        encode_kwargs={'normalize_embeddings': False}  # define se os embeddings devem ser normalizados (False mantém a escala original)
    )
    
    # cria um vetor de armazenamento a partir dos documentos fornecidos e do modelo de embeddings utilizando a classe chroma do langchain
    vectorstore = Chroma.from_documents(
        documents=splits,  # documentos que foram divididos em partes menores
        embedding=hf  # modelo de embeddings utilizado para transformar os documentos em vetores
    )

    # retorna o vetor de armazenamento como um retriever (objeto) para consultas posteriores
    return vectorstore.as_retriever()
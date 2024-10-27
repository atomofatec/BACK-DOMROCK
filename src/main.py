import os
import nltk
import spacy
from dotenv import load_dotenv
from scripts.doc_loader import load_data, load_and_chunk
from scripts.embeddings import generate_embeddings
from scripts.prompts import prompt_pull
from langchain_groq import ChatGroq
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# carrega as variáveis do arquivo .env
load_dotenv()

nltk.download('stopwords')  # baixa as stopwords
nlp = spacy.load('pt_core_news_sm')  # baixa o modelo nlp em pt-br do spacy 

# carrega e processa o .csv
file_path = load_data('data/chat_data.csv')  # caminho do .csv inicial (sem tratamento)

# carrega e faz o chunking dos documentos
splits = load_and_chunk(file_path)

# gera os embeddings e inicializa o retriever
retriever = generate_embeddings(splits)

# carrega o modelo de prompt
prompt = prompt_pull()

# inicializa o modelo llama com a chave de api do groq
llm = ChatGroq(model="llama3-8b-8192", api_key=os.getenv('GROQ_API_KEY'))

# formata os documentos
def format_docs(docs):
    return '\n\n'.join(doc.page_content for doc in docs)

# define a cadeia rag
def handle_query(question):
    # gera o contexto usando o retriever
    context_docs = retriever.invoke(question)
    formatted_context = format_docs(context_docs)
    
    # verifica se o contexto é relevante
    if not formatted_context or len(formatted_context.strip()) < 20:
        return "Como a pergunta não tem ligação com o contexto de análise de produtos, não sou capaz de responder."
    
    # cria o dicionário de entrada para a execução do modelo
    input_data = {
        'context': formatted_context,
        'question': question
    }
    
    # define a cadeia de execução com as informações de contexto e pergunta
    rag = (
        RunnablePassthrough()  # garante que o dicionário seja passado corretamente
        | (prompt + "\n\nComo informação adicional, responda em português.")
        | llm
        | StrOutputParser()
    )

    # executa a cadeia rag e retorna a resposta
    return rag.invoke(input_data)

# lê a pergunta do terminal
pergunta = input('Digite sua pergunta: ')

# executa a função handle_query
response = handle_query(pergunta)
print(response)
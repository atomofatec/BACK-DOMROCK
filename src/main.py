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
    context_docs = retriever.invoke(question) # busca documentos relevantes com base na pergunta
    formatted_context = format_docs(context_docs) # formata os documentos recuperados
    
    # verifica se o contexto é relevante
    # usa os documentos formatados anteriormente, caso tenha sido encontrado algum com base na pergunta
    # se o tamanho da string for maior que 20 caracteres, significa que encontrou um documento no contexto
    # se o tamanho for menor, significa que o contexto não é o suficiente
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
        | (prompt + "\n\nComo informação adicional, responda em português.\n\nNão peça desculpas pelo erro.\n\nEstamos no ano de 2024.\n\nQuando perguntado sobre gênero ou sexo, consultar a coluna reviewer_gender.\n\nQuando perguntado sobre desempenho ao longo do tempo ou em relação ao tempo, consultar a coluna submission_date.\n\nQuando perguntado sobre idade, ano de nascimento ou faixa etária, consultar a coluna reviewer_birth_year.\n\nIdentifique a coluna submission_date como a data em que o usuário publicou a review, e a coluna reviewer_birth_year como o ano de nascimento do usuário, não confunda as duas.")
        | llm
        | StrOutputParser()
    )

    # executa a cadeia rag e retorna a resposta
    return rag.invoke(input_data)

# loop para fazer perguntas
while True:
    # lê a pergunta do terminal
    pergunta = input('Digite sua pergunta (ou digite "Sair" para encerrar): ')
    
    # verifica se o usuário deseja sair
    if pergunta.strip().lower() == "sair":
        print("Encerrando o programa...")
        break
    
    # executa a função handle_query
    response = handle_query(pergunta)
    print(response)
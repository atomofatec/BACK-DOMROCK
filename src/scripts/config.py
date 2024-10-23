import os
from dotenv import load_dotenv

def load_env():
    load_dotenv() # carrega as variáveis de ambiente do .env
    
    # define variáveis de ambiente
    os.environ['LANGCHAIN_TRACING_V2'] = os.getenv('LANGCHAIN_TRACING_V2')  # chave de rastreamento do langchain
    os.environ['LANGCHAIN_API_KEY'] = os.getenv('LANGCHAIN_API_KEY')  # chave da api do langchain
    os.environ['GROQ_API_KEY'] = os.getenv('GROQ_API_KEY')  # chave da api do groq
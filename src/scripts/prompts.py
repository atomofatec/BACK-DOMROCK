from langchain import hub

def prompt_pull():
    # carrega um modelo de prompt do repositório do langchain hub
    return hub.pull('rlm/rag-prompt') # o modelo de prompt usado será o 'rlm/rag-prompt'
from transformers import pipeline

# modelo de geração de texto pré-treinado (gpt2)
generator = pipeline('text-generation', model='gpt2')

# função que gera o texto com base no comentário fornecido
# max_length define o tamanho da mensagem e num_return_sequences define quantidade de respostas
# necessário aprimoramento para respostas coerentes
def gerar_resposta(comentario):
    resposta = generator(f"Baseado neste comentário: '{comentario}', gere um resumo.", max_length=100, num_return_sequences=1)
    return resposta[0]['generated_text']

from transformers import pipeline

# modelo de geração de texto pré-treinado (gpt2)
generator = pipeline('text-generation', model='gpt2')

# função que gera o texto com base no comentário fornecido e retorna as notas relacionadas a ele
# max_length define o tamanho da mensagem e num_return_sequences define quantidade de respostas
# necessário aprimoramento para respostas coerentes
def gerar_resposta_por_produto(resultados):
    if not resultados:
        return "Nenhuma avaliação encontrada para o produto."

    resposta = f"Produto: {resultados['produto']}\n"
    resposta += f"Nota: {resultados['nota']}\n"
    resposta += f"Comentário: {resultados['comentário']}\n"
    
    return resposta
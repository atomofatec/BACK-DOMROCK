from transformers import pipeline

# modelo de geração de texto pré-treinado (gpt2)
generator = pipeline('text-generation', model='gpt2')

# função que gera o texto com base no comentário fornecido e retorna as notas relacionadas a ele
# max_length define o tamanho da mensagem e num_return_sequences define quantidade de respostas
# necessário aprimoramento para respostas coerentes
def gerar_resposta_por_produto(resultados):
    if not resultados:
        return "Nenhuma avaliação encontrada para o produto."
    
    nome_produto = resultados[0]['produto']  
    
    notas = [str(resultado['nota']) for resultado in resultados]
    
    notas_texto = ", ".join(notas)
    
    prompt = f"O produto {nome_produto} recebeu as seguintes notas de diferentes usuários: {notas_texto}. Gere um resumo dessas avaliações."
    
    resposta = generator(prompt, max_length=100, num_return_sequences=1)

    return resposta[0]['generated_text']

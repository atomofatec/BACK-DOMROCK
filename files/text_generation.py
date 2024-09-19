from transformers import pipeline

# Modelo de geração de texto pré-treinado (GPT-2)
generator = pipeline('text-generation', model='gpt2')

def gerar_resposta_por_produto(resultados):
    if not resultados:
        return "Nenhuma avaliação encontrada para o produto."
    
    nome_produto = resultados[0]['produto']
    
    notas = [str(resultado['nota']) for resultado in resultados]
    comentarios = [resultado['comentário'] for resultado in resultados]
    
    notas_texto = ", ".join(notas)
    comentarios_texto = " ".join(comentarios)
    
    prompt = (
        f"O produto '{nome_produto}' recebeu as seguintes notas de diferentes usuários: {notas_texto}. "
        f"Aqui estão alguns comentários dos usuários: {comentarios_texto}. "
        "Por favor, gere um resumo claro e conciso dessas avaliações, destacando os pontos positivos e negativos."
    )
    
    # Calcule o comprimento do prompt
    input_length = len(generator.tokenizer(prompt)['input_ids'])
    
    # Ajuste o max_new_tokens para garantir que a geração de texto possa ocorrer sem problemas
    resposta = generator(prompt, max_new_tokens=200, num_return_sequences=1)

    return resposta[0]['generated_text']

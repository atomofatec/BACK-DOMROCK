from transformers import pipeline, set_seed, AutoTokenizer

# Definindo a semente para reprodutibilidade
set_seed(52)

# Usando o pipeline com o modelo desejado
generator = pipeline('text-generation', model='egonrp/gpt2-medium-wikiwriter-squadv11-portuguese', pad_token_id=50256)
tokenizer = AutoTokenizer.from_pretrained('egonrp/gpt2-medium-wikiwriter-squadv11-portuguese')

def gerar_resposta_por_produto(resultados):
    if not resultados:
        return "Nenhuma avaliação encontrada para o produto."

    nome_produto = resultados[0]['produto']

    # Garantir que não haja duplicação de notas e comentários
    notas_unicas = list(dict.fromkeys([str(resultado['nota']) for resultado in resultados]))
    comentarios_unicos = list(dict.fromkeys([str(resultado['comentário']) for resultado in resultados]))

    # Limitar o número de caracteres dos comentários
    comentarios_unicos = [comentario[:200] for comentario in comentarios_unicos]

    # Transformar as notas e comentários únicos em texto
    notas_texto = ", ".join(notas_unicas)
    comentarios_texto = " ".join(comentarios_unicos)

    prompt = (
        f"O produto '{nome_produto}' recebeu as seguintes notas de diferentes usuários: {notas_texto}. "
        f"\n\n"
        f"Aqui estão alguns comentários dos usuários referentes ao produto {nome_produto}: {comentarios_texto}. "
        f"\n\n"
        f"Por favor, forneça um resumo claro e conciso das avaliações acima, destacando os pontos positivos e negativos sobre o {nome_produto}."
    )

    # Geração da resposta usando o pipeline generator
    output_sequences = generator(prompt, max_new_tokens=100, num_return_sequences=1)
    
    # Decode the generated text
    decoded_text = output_sequences[0]['generated_text']
    
    return decoded_text  # Return the generated text

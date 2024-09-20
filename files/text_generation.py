from transformers import pipeline

# Modelo de geração de texto pré-treinado (gpt2)
generator = pipeline('text-generation', model='gpt2')

def gerar_resposta_por_produto(resultados):
    # Caso não haja resultados, retorna uma mensagem de erro
    if not resultados:
        return "Nenhuma avaliação encontrada para o produto."
    
    # Nome do produto extraído do primeiro resultado
    nome_produto = resultados[0]['produto']
    
    # Variáveis para armazenar os comentários completos e contagem de avaliações positivas/negativas
    comentarios_completos = []
    total_positivos = 0
    total_negativos = 0
    
    # Listas para armazenar os aspectos positivos e negativos mencionados nos comentários
    aspectos_positivos = []
    aspectos_negativos = []
    
    # Mantendo a lógica de processamento dos resultados
    for resultado in resultados:
        if resultado['comentário']:
            comentario = resultado['comentário']
            data_submissao = resultado.get('data_submissão', 'Data não disponível')
            titulo_revisao = resultado.get('título_revisão', 'Sem título')
            nota = resultado.get('nota', 'Nota não disponível')
            categoria1 = resultado.get('site_category_lv1', 'Categoria 1 não disponível')
            categoria2 = resultado.get('site_category_lv2', 'Categoria 2 não disponível')

            recomenda_para_amigo = resultado.get('recomenda_para_amigo', 'Sem recomendação')
            
            # Verificar se a nota ou recomendação indicam um comentário positivo
            if nota >= 4 or 'Yes' in recomenda_para_amigo:
                total_positivos += 1
                aspectos_positivos.append(comentario)  # Armazena aspectos positivos
            else:
                total_negativos += 1
                aspectos_negativos.append(comentario)  # Armazena aspectos negativos
            
            # Monta o texto completo do comentário
            comentario_completo = (
                f"Título da Revisão: {titulo_revisao}\n"
                f"Nota: {nota}\n"
                f"Data de Submissão: {data_submissao}\n"
                f"Comentário: {comentario}\n"
                f"Recomendação para amigos: {recomenda_para_amigo}\n"
                f"Categoria 1: {categoria1}\n"
                f"Categoria 2: {categoria2}"
            )
            comentarios_completos.append(comentario_completo)  # Adiciona o comentário completo à lista
    
    # Limitar a 3 comentários completos para exibição
    comentarios_texto = " \n\n ".join(comentarios_completos[:3])
    
    # Resumo geral sobre os comentários
    resumo_geral = (
        f"O produto {nome_produto} recebeu um total de {len(resultados)} avaliações. "
        f"Dessas, {total_positivos} foram positivas e {total_negativos} foram negativas."
    )
    
    # Resumo escrito baseado nos aspectos extraídos
    if aspectos_positivos:
        resumo_positivo = f"Os clientes elogiaram aspectos como: {', '.join(aspectos_positivos[:2])}."
    else:
        resumo_positivo = "Não houve aspectos positivos mencionados."

    if aspectos_negativos:
        resumo_negativo = f"Por outro lado, as principais reclamações foram: {', '.join(aspectos_negativos[:2])}."
    else:
        resumo_negativo = "Não houve aspectos negativos mencionados."

    # Gera o prompt com as informações completas
    prompt = (
        f"{resumo_geral}\n\n"
        f"{resumo_positivo}\n"
        f"{resumo_negativo}\n\n"
        f"Aqui estão alguns comentários:\n\n{comentarios_texto}."
    )

    print("Prompt gerado:", prompt)  # Verificando o prompt
    
    # MAX_LENGTH: Define o comprimento máximo da resposta gerada
    # NUM_RETURN_SEQUENCES: Gera uma única sequência de texto como resposta
    # TEMPERATURE: Controla a criatividade da resposta; quanto maior o valor, mais aleatória e variada será a geração.
    # TOP_K: Limita a escolha das próximas palavras às 50 mais prováveis, garantindo diversidade na resposta.
    # TRUNCATION (True): Garante que o prompt não exceda o comprimento máximo permitido pelo modelo; o texto será cortado para se ajustar ao limite se necessário.  
    # PROMPT: Texto de entrada que o modelo usará para gerar a resposta

    # Gera a resposta final baseada no prompt
    resposta = generator(prompt, max_length=900, num_return_sequences=1, temperature=0.9, top_k=50, truncation=True)
    
    return resposta[0]['generated_text']

    # A função lê os dados de outra fonte e mantém a lógica dinâmica

    
    
   
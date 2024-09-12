# Gera uma resposta com base nas notas encontradas para o produto
def gerar_resposta_por_produto(resultados):
    if not resultados:
        return "Nenhuma avaliação encontrada para esse produto."
    
    produto_nome = resultados[0]['produto']  # Nome do produto
    notas = [r['nota'] for r in resultados]  # Lista de notas
    notas_str = ', '.join(map(str, notas))  # Formata as notas como string
    
    resposta = (f"O produto '{produto_nome}' recebeu as seguintes notas: {notas_str} "
                f"de diferentes usuários.")
    
    return resposta

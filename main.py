from files.preprocess import load_and_preprocess_data
from files.embeddings import gerar_embeddings
from files.faiss_search import criar_faiss_index, buscar_por_produto
from files.text_generation import gerar_resposta_por_produto
from sentence_transformers import SentenceTransformer
import pandas as pd

def gerar_resposta_por_produto(resultados):
    if not resultados:  # Se resultados for um dicionário vazio
        return "Nenhuma avaliação encontrada para o produto solicitado."

    produto = resultados.get('product_name', 'Produto desconhecido')
    nota = resultados.get('overall_rating', 'Nota não disponível')
    comentario = resultados.get('review_text', None)

    # Verifica se o comentário está ausente ou é 'nan'
    if comentario is None or pd.isna(comentario):
        comentario = 'Nenhum comentário disponível.'

    resposta = f"Produto: {produto}\nNota: {nota}\nComentário: {comentario}"
    return resposta


def buscar_por_produto(df, consulta_produto):
    # Converte a consulta para minúsculas e remove espaços
    consulta_produto = consulta_produto.lower().strip()

    # Verifica se a consulta é uma palavra inteira em algum nome de produto no DataFrame
    resultado = df[df['product_name_processado'].str.lower().str.contains(r'\b' + consulta_produto + r'\b')]

    if not resultado.empty:
        return resultado.iloc[0].to_dict()  # Retorna como dicionário do primeiro resultado
    else:
        return {}


def main():
    # Carregar e pré-processar os dados
    df = load_and_preprocess_data(r'data\chat_data.csv')

    model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

    # Definir as colunas de texto para geração de embeddings
    text_columns = ['product_name_processado', 'product_brand_processado', 'site_category_lv1_processado', 
                    'site_category_lv2_processado', 'review_title_processado', 'recommend_to_a_friend_processado', 
                    'review_text_processado', 'reviewer_gender_processado', 'reviewer_state_processado']

    # Gerar embeddings
    embeddings_dict = gerar_embeddings(df, text_columns)

    # Criar índice faiss para os embeddings
    index = criar_faiss_index(embeddings_dict['review_text_processado'])

    # Comentar ou remover a linha abaixo
    # print("Nomes dos produtos disponíveis:", df['product_name_processado'].tolist())
    print("Bem-vindo ao Chat ATM! Digite 'sair' para encerrar.")

    while True:
        consulta_produto = input("Qual produto você deseja receber a avaliação? ")

        if consulta_produto.lower() == 'sair':
            print("Encerrando o Chat ATM. Até mais!")
            break

        # Consultar por comentário similar
        resultados = buscar_por_produto(df, consulta_produto)

        # Gerar resposta
        resposta_gerada = gerar_resposta_por_produto(resultados)
        print(f"Resposta gerada: {resposta_gerada}")

if __name__ == "__main__":
    main()
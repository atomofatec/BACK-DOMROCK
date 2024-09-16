from files.preprocess import load_and_preprocess_data
from files.embeddings import gerar_embeddings
from files.faiss_search import criar_faiss_index, buscar_por_produto
from files.text_generation import gerar_resposta_por_produto
from sentence_transformers import SentenceTransformer

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

    print("Bem-vindo ao chatbot! Digite 'sair' para encerrar.")

    while True:
        consulta_produto = input("Qual a sua dúvida? ")

        if consulta_produto.lower() == 'sair':
            print("Encerrando o chatbot. Até mais!")
            break

        # Consultar por comentário similar
        resultados = buscar_por_produto(df, index, consulta_produto, model, k=10)

        # Gerar resposta com GPT-2
        if resultados:
            resposta_gerada = gerar_resposta_por_produto(resultados)
            print(f"Resposta gerada: {resposta_gerada}")
        else:
            print("Produto não encontrado ou sem avaliações.")

if __name__ == "__main__":
    main()

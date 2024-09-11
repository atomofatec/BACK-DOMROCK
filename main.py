import preprocess
import embeddings
import faiss_search
import text_generation

# criar ambiente virtual em python (opcional)
# instalar os pacotes com o comando abaixo (obrigatório):
# pip install -r requirements.txt

# carregar e pré-processar os dados
df = preprocess.load_and_preprocess_data('chat_data.csv')

# gerar embeddings
text_columns = ['product_name_processado', 'product_brand_processado', 'site_category_lv1_processado', 
                'site_category_lv2_processado', 'review_title_processado', 'recommend_to_a_friend_processado', 
                'review_text_processado', 'reviewer_gender_processado', 'reviewer_state_processado']
embeddings_dict = embeddings.gerar_embeddings(df, text_columns)

# criar índice faiss para os embeddings
index = faiss_search.criar_faiss_index(embeddings_dict['review_text_processado'])

# consultar por comentário similar (o comentário está sendo passado diretamente no código, criar interface no prompt para interação)
consulta = "Produto bom."
resultados = faiss_search.buscar_no_faiss(df, index, consulta, embeddings.model, k=1)

# gerar resposta com gpt2
if resultados:
    comentario_proximo = resultados[0]['comentário']
    mensagem_gerada = text_generation.gerar_resposta(comentario_proximo)
    print(f"Comentário mais similar: {comentario_proximo}")
    print(f"Mensagem gerada: {mensagem_gerada}")
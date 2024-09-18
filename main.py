from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pandas as pd
from datasets import load_dataset
import re
from transformers import pipeline
from files.embeddings import gerar_embeddings
from files.faiss_search import criar_faiss_index, buscar_no_faiss
from files.preprocess import preprocess_text

# Inicializa o modelo de embeddings
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# Função para limpar o texto das avaliações
def clean_text(text):
    text = re.sub(r'[^\w\s]', '', text)  
    return text.lower()

# Carregar o dataset
dataset = load_dataset('ruanchaves/b2w-reviews01')
df = pd.DataFrame(dataset['train'])

# Aplicar a função de limpeza nas avaliações
df['review_text'] = df['review_text'].fillna('') 
df['review_text'] = df['review_text'].apply(clean_text)

# Verificar valores nulos 'overall_rating' e 'product_name'
df = df[df['overall_rating'].notnull() & df['product_name'].notnull()]

# Agrupar 'site_category_lv1' e 'product_name' pela média de avaliações e contagem de comentários
grouped = df.groupby(['site_category_lv1', 'product_name']).agg(
    avg_rating=('overall_rating', 'mean'),
    review_count=('review_text', 'count')
).reset_index()

# Classificar produtos/categorias pela avaliação média
grouped = grouped.sort_values(['site_category_lv1', 'avg_rating'], ascending=[True, False])

# Selecionar os produtos mais bem avaliados por categoria
top_products_per_category = grouped.groupby('site_category_lv1').head(1)

# Exibir categorias e produtos mais bem avaliados
print(top_products_per_category[['site_category_lv1', 'product_name', 'avg_rating', 'review_count']])

# Adicionar os produtos mais bem avaliados ao dataframe principal
df = df.merge(top_products_per_category[['site_category_lv1', 'product_name']],
              on=['site_category_lv1', 'product_name'],
              how='left',
              indicator=True)

# Manter apenas os produtos que estão entre os mais bem avaliados
df = df[df['_merge'] == 'both'].drop(columns=['_merge'])

# Gerar embeddings para as colunas relevantes
text_columns = ['product_name', 'review_text', 'site_category_lv1']
embeddings_dict = gerar_embeddings(df, text_columns)

# Criar índice FAISS para os embeddings de review_text
index = criar_faiss_index(np.array(embeddings_dict['review_text']))

# Inicializa o pipeline de geração de texto com o modelo GPT-2
generator = pipeline('text-generation', model='gpt2')

# Função que gera uma resposta com base no comentário fornecido
def gerar_resposta(comentario):
    resposta = generator(
        f"Baseado neste comentário: '{comentario}', gere um resumo.",
        max_new_tokens=100,  # Limita o comprimento da nova geração
        truncation=True,    
        pad_token_id=generator.tokenizer.eos_token_id  # Define o ID do token de preenchimento
    )
    return resposta[0]['generated_text']

# Consultar por múltiplos comentários associados ao produto
consulta_produto = "Copo Acrílico Com Canudo 500ml Rocie"
resultados_produto = buscar_no_faiss(df, index, consulta_produto, model, k=10)

# Gerar resposta com base nas avaliações do produto
if resultados_produto:
    comentarios = [resultado['comentário'] for resultado in resultados_produto]
    respostas_geradas = [gerar_resposta(comentario) for comentario in comentarios]
    
    # Exibir respostas geradas
    for resposta in respostas_geradas:
        print(f"Resposta gerada: {resposta}")
else:
    print("Produto não encontrado ou sem avaliações.")

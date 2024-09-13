import re
import pandas as pd
from datasets import load_dataset
from embeddings import gerar_embeddings  
from faiss_search import criar_faiss_index, buscar_no_faiss  
from sentence_transformers import SentenceTransformer
import numpy as np

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

# Verificar se há valores nulos nas colunas 'overall_rating' e 'product_name'
df = df[df['overall_rating'].notnull() & df['product_name'].notnull()]

# Agrupar por 'site_category_lv1' e 'product_name' 
grouped = df.groupby(['site_category_lv1', 'product_name']).agg(
    avg_rating=('overall_rating', 'mean'), 
    review_count=('review_text', 'count')  
).reset_index()

# Classificar produtos por categoria e avaliação média
grouped = grouped.sort_values(['site_category_lv1', 'avg_rating'], ascending=[True, False])

# Selecionar os produtos mais bem avaliados por categoria
top_products_per_category = grouped.groupby('site_category_lv1').head(1)

# Exibir as categorias e seus produtos mais bem avaliados
print(top_products_per_category[['site_category_lv1', 'product_name', 'avg_rating', 'review_count']])

# Adicionar os produtos mais bem avaliados ao dataframe principal
df = df.merge(top_products_per_category[['site_category_lv1', 'product_name']], 
              on=['site_category_lv1', 'product_name'], 
              how='left', 
              indicator=True)

# Manter apenas os produtos que estão entre os mais bem avaliados
df = df[df['_merge'] == 'both'].drop(columns=['_merge'])

# Gerar embeddings para as colunas relevantes incluindo os produtos mais bem avaliados
text_columns = ['product_name', 'review_text', 'site_category_lv1']
embeddings_dict = gerar_embeddings(df, text_columns)

# Criar índice FAISS para os embeddings de review_text
index = criar_faiss_index(np.array(embeddings_dict['review_text']))

# Consultar por comentário similar
consulta = "Produto excelente, ótima qualidade."
resultados = buscar_no_faiss(df, index, consulta, model, k=1)

# Exibir resultados
if resultados:
    comentario_proximo = resultados[0]['comentário']
    produto_proximo = resultados[0]['produto']
    print(f"Comentário mais similar: {comentario_proximo}")
    print(f"Produto: {produto_proximo}")

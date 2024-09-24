import pandas as pd
import spacy

# Instalar o modelo com o comando abaixo
# python -m spacy download pt_core_news_sm

# Carregar o modelo de linguagem do spaCy
nlp = spacy.load('pt_core_news_sm')

# Função para processar o texto e retornar normalizado
def preprocess_text(text):
    if pd.isna(text):
        return ''
    doc = nlp(text)
    tokens = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct]
    return " ".join(tokens)

# Função que carrega o csv e pré-processa os dados
def load_and_preprocess_data(file_path):
    df = pd.read_csv(file_path)
    text_columns = ['product_name', 'product_brand', 'site_category_lv1', 'site_category_lv2', 
                    'review_title', 'recommend_to_a_friend', 'review_text', 'reviewer_gender', 
                    'reviewer_state']
    
    for col in text_columns:
        df[f'{col}_processado'] = df[col].apply(preprocess_text)
    
    return df

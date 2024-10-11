import nltk
import pandas as pd
import re
import spacy
from nltk.corpus import stopwords

# carrega o arquivo com os dados
df = pd.read_csv(r'data\chat_data.csv')

# exibe algumas informações do dataframe
print("Informações iniciais do DataFrame:")
print(df.info())

# remove as linhas onde os campos 'product_id', 'review_text' e 'product_name' estão vazios
df.dropna(subset=['product_id', 'review_text', 'product_name' ], inplace=True)

# inicia o modelo SpaCy para lematização dos dados
nlp = spacy.load("pt_core_news_sm")  # usa o modelo adapatado para pt-br

# baixa as stopwords em pt-br do NLTK
nltk.download('stopwords')
stop_words = set(stopwords.words('portuguese'))

# função de normalização do texto
def normalize_text(text):
    text = re.sub(r'[^\w\s]', '', text)  # remove as pontuações
    text = re.sub(r'\d+', '', text)  # remove os números
    text = text.lower()  # transforma tudo em minúsculo
    return text

# função de lematização e remoção de stop words (testar stemização e outras estratégias)
def lemmatize_and_remove_stopwords(text):
    doc = nlp(text)
    return ' '.join([token.lemma_ for token in doc if token.lemma_ not in stop_words])

# função que divide o texto em sentenças (testar outras estratégias, como divisão por palavras ou parágrafos)
def divide_text_in_topics(text):
    return text.split('. ')  # divide usando ponto e espaço


# aplicar as funções de limpeza e normalização na coluna 'review_text'
df['review_text'] = df['review_text'].apply(normalize_text)  # normalização
df['review_text'] = df['review_text'].apply(
    lemmatize_and_remove_stopwords)  # lematização e remoção de stop words
df['review_text'] = df['review_text'].apply(
    lambda x: divide_text_in_topics(x))  # divisão em sentenças

# converte a data de submissão da review para um formato adequado
df['submission_date'] = pd.to_datetime(df['submission_date'])

# remove as colunas que não serão usadas para exigir menos do modelo na hora de processar
colunas_para_remover = ['reviewer_id', 'recommend_to_a_friend']
df.drop(columns=colunas_para_remover, inplace=True)

# printa as informações finais do dataframe, com os dados limpos
print("Informações do DataFrame após limpeza:")
print(df.info())

# exporta o dataframe limpo para um novo arquivo .csv
df.to_csv(r'data\chat_data_processado.csv', index=False)

print("Arquivo limpo salvo como 'data_preprocessed.csv'.")
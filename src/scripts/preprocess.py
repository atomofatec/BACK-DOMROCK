import re
import pandas as pd
from unidecode import unidecode
from nltk.corpus import stopwords

# baixa as stopwords em pt-br
stopwords_list = set(stopwords.words('portuguese'))

def normalize_text(text):
    # deixa tudo minúsculo
    text = text.lower()
    
    text = unidecode(text) # remove os acentos codificados

    # remove espaços desnecessários
    text = re.sub(r'\s+', ' ', text).strip()
    
    # remove pontuação
    text = re.sub(r'[^\w\s]', '', text)
    
    return text

def remove_stopwords(text):
    return ' '.join([word for word in text.split() if word not in stopwords_list]) # remove as palavras do texto que estiverem na lista de stopwords

def format_birth_year(year):
    """Formata o ano de nascimento, removendo a parte decimal e lidando com NaN."""
    if pd.isna(year):  # Verifica se é NaN
        return None
    return str(int(float(year))) if year else None
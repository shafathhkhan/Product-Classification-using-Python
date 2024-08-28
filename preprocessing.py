import pandas as pd
import string

def clean_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text

def load_and_preprocess_data(file_path):
    data = pd.read_csv(file_path)
    data.drop(columns=['Unnamed: 0'], inplace=True)
    data.dropna(inplace=True)
    data['processed_name'] = data['ProductName'].apply(clean_text)
    data['processed_description'] = data['Description'].apply(clean_text)
    return data

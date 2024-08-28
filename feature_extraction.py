from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack
import joblib

def extract_features(df, fit=True):
    if fit:
        vectorizer_name = TfidfVectorizer(max_features=5000)
        vectorizer_description = TfidfVectorizer(max_features=5000)
        
        X_name = vectorizer_name.fit_transform(df['processed_name'])
        X_description = vectorizer_description.fit_transform(df['processed_description'])
        
        # Save the vectorizers
        joblib.dump(vectorizer_name, 'vectorizer_name.pkl')
        joblib.dump(vectorizer_description, 'vectorizer_description.pkl')
    else:
        vectorizer_name = joblib.load('vectorizer_name.pkl')
        vectorizer_description = joblib.load('vectorizer_description.pkl')
        
        X_name = vectorizer_name.transform(df['processed_name'])
        X_description = vectorizer_description.transform(df['processed_description'])
    
    X = hstack([X_name, X_description])
    
    if fit:
        y = df['Category']  # or df['SubCategory']
        return X, y
    else:
        return X

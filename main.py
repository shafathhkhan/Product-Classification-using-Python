import pandas as pd
import joblib
from preprocessing import load_and_preprocess_data, clean_text
from feature_extraction import extract_features
from sklearn.linear_model import LogisticRegression

def predict_category(model, product_name, product_description):
    data = {
        'processed_name': [clean_text(product_name)],
        'processed_description': [clean_text(product_description)]
    }
    df = pd.DataFrame(data)
    
    X_new = extract_features(df, fit=False)
    predicted_category = model.predict(X_new)
    return predicted_category

def main():
    # Load and preprocess the training data
    df = load_and_preprocess_data("Dataset_nlp-1.csv")
    
    # Extract features and labels
    X, y = extract_features(df, fit=True)
    
    # Train the model
    model = LogisticRegression()
    model.fit(X, y)
    
    # Save the trained model
    joblib.dump(model, 'model.pkl')
    
    # Load the model for prediction
    model = joblib.load('model.pkl')
    
    product_name = input("Enter the product name: ")
    product_description = input("Enter the product description: ")
    
    predicted_category = predict_category(model, product_name, product_description)
    print(f"Predicted Category: {predicted_category}")

if __name__ == "__main__":
    main()

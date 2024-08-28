# Product Category Prediction Model

This repository contains a machine learning pipeline to predict product categories based on product names and descriptions. The model is built using TF-IDF vectorization and a logistic regression classifier.

## Requirements

- Python 3.6 or higher
- pandas
- scikit-learn
- scipy
- joblib

## Installation

First, ensure you have Python installed on your machine. Then, install the required Python packages using pip:

```sh
pip install pandas scikit-learn scipy joblib


Usage

1. Preprocess and Train the Model
To preprocess the data and train the model, follow these steps:

Save your dataset CSV file (e.g., Dataset_nlp 1.csv) in an accessible location.
Modify the file path in the main.py script to point to your dataset.
Run the main.py script to preprocess the data, extract features, train the model, and save the vectorizers and model to disk.

2. Predict Product Categories
After training the model, you can use it to predict the category of new products. The main.py script also includes functionality for predicting the category of a single product based on its name and description.

To predict the category:

Ensure the trained model and vectorizers are saved (this happens automatically after running the training phase).
Run the main.py script again and provide the product name and description when prompted

Example Input:
    Enter the product name: Apple iPhone 13 Pro
    Enter the product description: The iPhone 13 Pro features a sleek design with a durable ceramic shield. It comes with a powerful A15 Bionic chip and a Pro camera system for advanced photography. Enjoy the Super Retina XDR display and long-lasting battery life.

Example Output: 
    Predicted Category: Electronics


File Structure
preprocessing.py: Contains functions for loading and preprocessing data.
feature_extraction.py: Contains functions for extracting features using TF-IDF vectorization.
main.py: Main script to preprocess data, train the model, and predict product categories.

Functions
preprocessing.py
clean_text(text): Cleans and preprocesses the input text.
load_and_preprocess_data(file_path): Loads and preprocesses the dataset from the given file path.
feature_extraction.py
extract_features(df, fit=True): Extracts features from the DataFrame and fits vectorizers if fit=True. Otherwise, transforms using pre-fitted vectorizers.
main.py
predict_category(model, product_name, product_description): Predicts the category of a product based on its name and description.
main(): Main function to execute the training and prediction workflow.

Troubleshooting
Ensure the dataset CSV file path is correctly specified.
Make sure all required packages are installed.
Check for any preprocessing issues that may result in empty text after cleaning.    


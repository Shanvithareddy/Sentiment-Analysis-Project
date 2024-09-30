
# Sentiment Analysis Project

## Overview
This project implements a **Sentiment Analysis** model using machine learning techniques to classify text reviews as **positive** or **negative**. The goal is to analyze user reviews or feedback and automatically determine their sentiment. This project is built using Python and leverages Natural Language Processing (NLP) techniques.

## Features
- Preprocessing of text data (tokenization, stopword removal, vectorization)
- Model training and evaluation using machine learning algorithms
- Sentiment classification into positive or negative
- Visualization of results (optional)

## Dataset
The dataset used for this project is the **IMDb Movie Reviews Dataset**, which contains 5,000 movie reviews labeled as either positive or negative.

## Technology Stack
- **Python** for the implementation
- **scikit-learn** for machine learning models
- **NLTK** for text preprocessing (stopwords, tokenization)
- **Pandas & NumPy** for data handling and manipulation
- **Matplotlib & Seaborn** for data visualization

## Workflow
1. **Data Loading**: Load the dataset into a pandas DataFrame.
2. **Text Preprocessing**: Tokenize the text, remove stopwords, and vectorize the text data using **CountVectorizer** or **TF-IDF**.
3. **Model Training**: Train a machine learning model (e.g., **Multinomial Naive Bayes**, **Logistic Regression**, or **SVM**).
4. **Evaluation**: Evaluate the model using accuracy, precision, recall, and confusion matrix.
5. **Prediction**: Use the trained model to predict sentiment on new text inputs.

## Installation

1. Clone this repository:
    ```bash
    git clone https://github.com/Shanvithareddy/Sentiment-Analysis-Project.git
    ```
2. Install required libraries:
    ```bash
    pip install -r requirements.txt
    ```

## Usage
1. Preprocess the dataset using text cleaning functions.
2. Train the model on the preprocessed data.
3. Test the model on the test dataset.
4. Predict sentiment for new input reviews.

## How to Run

1. Download the dataset and place it in the `data/` directory.
2. Run the following command to train the model:
    ```bash
    python movie_reviews_sentiment_analysis.py
    ```
3. To predict a new review:
    ```bash
    python predict.py "Your movie review text here"
    ```

## Results
- The model achieved an accuracy of **85%** on the IMDb movie review dataset.


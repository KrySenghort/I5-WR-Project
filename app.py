import streamlit as st
import pickle
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Load the model and vectorizer
def load_model_and_vectorizer(model_path="formal_svm_model.pkl", vectorizer_path="formal_vectorizer.pkl"):
    with open(model_path, 'rb') as model_file:
        svm_model = pickle.load(model_file)
    with open(vectorizer_path, 'rb') as vectorizer_file:
        vectorizer = pickle.load(vectorizer_file)
    return svm_model, vectorizer

# Preprocessing functions
def lowercase_text(text):
    return text.lower()

def tokenize_text(text):
    return word_tokenize(text)

def remove_special_characters(tokens):
    return [re.sub(r'[^a-zA-Z0-9]', '', token) for token in tokens if token.isalnum()]

def remove_stopwords_and_punctuation(tokens):
    stop_words = set(stopwords.words('english'))
    return [token for token in tokens if token not in stop_words]

def remove_urls(text):
    return re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)

def stem_tokens(tokens):
    stemmer = PorterStemmer()
    return [stemmer.stem(token) for token in tokens]

# Feature extraction function
def preprocess_and_transform(email_text, vectorizer=None):
    email_text = lowercase_text(email_text)
    tokens = tokenize_text(email_text)
    tokens = remove_special_characters(tokens)
    tokens = remove_stopwords_and_punctuation(tokens)
    email_text = remove_urls(' '.join(tokens))
    tokens = tokenize_text(email_text)
    stemmed_tokens = stem_tokens(tokens)
    preprocessed_text = ' '.join(stemmed_tokens)

    # Use vectorizer to extract features
    if vectorizer:
        return vectorizer.transform([preprocessed_text])
    else:
        # Create vectorizer if none exists
        return extract_features([preprocessed_text])[0]

# Streamlit UI
def main():
    st.title("Spam Email Classifier")
    st.write("""
    This app classifies whether an email is spam or not spam based on its content.
    Please enter the email content below:
    """)

    # User input
    email_text = st.text_area("Enter Email Text:")

    if st.button("Predict"):
        if email_text:
            # Load model and vectorizer
            svm_model, vectorizer = load_model_and_vectorizer()

            # Preprocess the email text and extract features
            email_features = preprocess_and_transform(email_text, vectorizer)

            # Make prediction
            prediction = svm_model.predict(email_features)

            # Display result
            if prediction[0] == 1:
                st.write("**Prediction**: This email is **SPAM**.")
            else:
                st.write("**Prediction**: This email is **NOT SPAM**.")
        else:
            st.write("Please enter an email text to classify.")

# Run the app
if __name__ == "__main__":
    main()

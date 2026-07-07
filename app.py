import streamlit as st
import pickle
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

# Ensure NLTK resources are available in the app environment
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('punkt_tab')

# Load the saved model and vectorizer
with open('logistic_regression_imdb.pkl', 'rb') as f:
    model = pickle.load(f)

with open('tfidf_vectorizer.pkl', 'rb') as f:
    tfidf = pickle.load(f)

# Preprocessing function
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    # Tokenize and lowercase
    tokens = word_tokenize(text.lower())
    # Remove non-alphabetic and stopwords, then lemmatize
    processed = [lemmatizer.lemmatize(w) for w in tokens if w.isalpha() and w not in stop_words]
    return " ".join(processed)

# Streamlit UI
st.title("IMDB Movie Sentiment Analysis")
st.write("Enter a movie review below to predict if it is Positive or Negative.")

user_input = st.text_area("Review:", "")

if st.button("Predict"):
    if user_input.strip() != "":
        clean_text = preprocess_text(user_input)
        vectorized_text = tfidf.transform([clean_text])
        prediction = model.predict(vectorized_text)[0]
        
        if prediction == "positive":
            st.success(f"The sentiment is: **{prediction.upper()}**")
        else:
            st.error(f"The sentiment is: **{prediction.upper()}**")
    else:
        st.warning("Please enter some text.")

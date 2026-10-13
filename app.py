```python
import streamlit as st
import joblib

st.set_page_config(
    page_title="Fake News Detector",
    page_icon="🔎",
    layout="wide"
)

@st.cache_resource
def load_models():
    model = joblib.load("fake_news_model.pkl")
    vectorizer = joblib.load("tfidf_vectorizer.pkl")
    return model, vectorizer

st.title("🔎 Fake News Detection")
st.write("### Fake News Detection using Python and Machine Learning")
st.write("Enter a news article below to check its prediction.")

news = st.text_area(
    "Enter news text here:",
    height=200,
    placeholder="Paste your news headline or article here..."
)

if st.button("Check News"):
    if news.strip() == "":
        st.warning("Please enter some news text first.")
    else:
        try:
            model, vectorizer = load_models()
            news_vector = vectorizer.transform([news])
            prediction = model.predict(news_vector)[0]

            st.subheader("Prediction Result")

            if str(prediction) == "0":
                st.error("🚨 This news is predicted to be FAKE.")
            elif str(prediction) == "1":
                st.success("✅ This news is predicted to be REAL.")
            else:
                st.info(f"Model prediction: {prediction}")

            st.caption(
                "This is a machine-learning prediction, not a guarantee. "
                "Please verify important news using reliable sources."
            )

        except Exception as e:
            st.error("Something went wrong while checking the news.")
            st.code(str(e))

st.divider()

st.subheader("About This Project")
st.write(
    "This project uses Natural Language Processing (NLP), "
    "TF-IDF vectorization, and a trained machine-learning model "
    "to classify news text."
)

st.write("**Technologies:** Python, Streamlit, Scikit-learn, TF-IDF, Joblib")
```

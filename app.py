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

st.markdown("""
<style>
.stApp {
    background-color: #f4f8fd;
}
.hero {
    padding: 32px;
    border-radius: 16px;
    background: linear-gradient(120deg, #e3efff, #ffffff);
    border: 1px solid #d5e5fa;
}
.hero h1 {
    color: #12325a;
}
.hero p {
    color: #415a77;
    font-size: 18px;
}
</style>
""", unsafe_allow_html=True)

st.sidebar.title("🔎 Fake News Detector")
page = st.sidebar.radio(
    "Navigate",
    ["Home", "Check News", "About"]
)

if page == "Home":
    st.markdown("""
    <div class="hero">
        <h1>Detect Fake News with Machine Learning</h1>
        <p>Analyze news text using your trained machine-learning model.</p>
    </div>
    """, unsafe_allow_html=True)

    st.write("")
    col1, col2, col3 = st.columns(3)

    col1.info("🛡️ Machine-learning powered")
    col2.info("⚡ Quick predictions")
    col3.info("✍️ Simple to use")

elif page == "Check News":
    st.title("📰 Check a News Article")

    news = st.text_area(
        "Paste a headline or news article",
        height=220,
        placeholder="Type or paste news text here..."
    )

    if st.button("🔍 Check News", type="primary"):
        if not news.strip():
            st.warning("Please enter some news text.")
        else:
            try:
                model, vectorizer = load_models()
                features = vectorizer.transform([news])
                prediction = model.predict(features)[0]

                if str(prediction) == "0":
                    st.error("🚨 Likely Fake News")
                elif str(prediction) == "1":
                    st.success("✅ Likely Real News")
                else:
                    st.info(f"Model prediction: {prediction}")

                st.caption(
                    "This is a model prediction, not proof. "
                    "Verify important claims using reliable sources."
                )

            except Exception as error:
                st.error("Prediction failed.")
                st.code(str(error))

elif page == "About":
    st.title("📘 About This Project")

    st.markdown("""
    ## Fake News Detection with Python and Machine Learning

    ### 🎯 Project Objective
    This project uses machine learning and Natural Language
    Processing (NLP) to classify news text as likely fake
    or likely real.

    ### ⚙️ How Does It Work?
    1. The user enters a headline or news article.
    2. TF-IDF converts the text into numerical features.
    3. The trained model analyzes the features.
    4. The website displays its prediction.

    ### 🛠️ Technologies Used
    - **Python:** Programming language
    - **Streamlit:** Interactive website
    - **Scikit-learn:** Machine-learning tools
    - **TF-IDF:** Text feature extraction
    - **Joblib:** Loading saved model files

    ### 🌟 Key Features
    - Simple and user-friendly interface
    - News headline and article analysis
    - Fast model predictions
    - Accessible through a web browser

    ### 🎓 Project Information
    **Project Name:** Fake News Detection with Python
    and Machine Learning

    **Application:** Fake News Detector

    **Purpose:** College project

    ### ⚠️ Important Disclaimer
    The prediction is not a guarantee that an article is true
    or false. Verify important claims using reliable,
    independent sources.
    """)

st.divider()
st.caption("Fake News Detector | Python & Machine Learning")
```

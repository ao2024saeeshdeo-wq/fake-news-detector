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
.stApp { background-color: #f4f8fd; }
.hero {
    padding: 32px;
    border-radius: 16px;
    background: linear-gradient(120deg, #e3efff, #ffffff);
    border: 1px solid #d5e5fa;
}
.hero h1 { color: #12325a; }
</style>
""", unsafe_allow_html=True)

st.sidebar.title("🔎 Fake News Detector")
page = st.sidebar.radio("Navigate", ["Home", "Check News"])

if page == "Home":
    st.markdown("""
    <div class="hero">
        <h1>Detect Fake News with Machine Learning</h1>
        <p>Analyze news text using your trained machine-learning model.</p>
    </div>
    """, unsafe_allow_html=True)

    st.write("")
    a, b, c = st.columns(3)
    a.info("🛡️ Machine-learning powered")
    b.info("⚡ Quick predictions")
    c.info("✍️ Simple to use")

    if st.button("Get Started"):
        st.info("Choose Check News in the sidebar to begin.")

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
                    "Verify important claims with reliable sources."
                )
            except FileNotFoundError:
                st.error(
                    "Model files not found. Upload both .pkl files "
                    "to the same folder as app.py."
                )
            except Exception as error:
                st.error("Prediction failed. Check the model files.")
                st.code(str(error))


st.divider()
st.caption("Fake News Detector | College Project")

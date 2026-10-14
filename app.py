import streamlit as st
import joblib

# 1. Page Configuration
st.set_page_config(
    page_title="Fake News Detector",
    page_icon="🔍",
    layout="wide"
)

# 2. Model Loading
@st.cache_resource
def load_models():
    model = joblib.load("fake_news_model.pkl")
    vectorizer = joblib.load("tfidf_vectorizer.pkl")
    return model, vectorizer

# 3. Custom CSS Styling
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

# 4. Sidebar Navigation
st.sidebar.title("🔍 Fake News Detector")
page = st.sidebar.radio("Navigate", ["Home", "Check News"])

# 5. Home Page
if page == "Home":
    st.markdown("""
    <div class="hero">
        <h1>Detect Fake News with Machine Learning</h1>
        <p>Analyze news text using your trained machine-learning model.</p>
    </div>
    """, unsafe_allow_html=True)

    st.write("")
    a, b, c = st.columns(3)
    a.info("💡 Machine-learning powered")
    b.info("⚡ Quick predictions")
    c.info("🔥 Simple to use")

    if st.button("Get Started"):
        st.info("Choose Check News in the sidebar to begin.")

# 6. Check News Page
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
            # Check length to warn about TF-IDF limits on short text
            word_count = len(news.split())
            if word_count < 10:
                st.warning("⚠️ Input text is very short. TF-IDF models require full paragraphs or longer context for reliable accuracy.")

            try:
                model, vectorizer = load_models()
                features = vectorizer.transform([news])
                prediction = model.predict(features)[0]

                pred_str = str(prediction).strip().lower()

                # Display model output value
                st.caption(f"🔍 Raw Model Output Value: `{prediction}`")

                # Correct mapping: 0 -> FAKE, 1 -> REAL
                if pred_str in ["0", "0.0", "fake"]:
                    st.error("🚨 Likely Fake News")
                elif pred_str in ["1", "1.0", "real"]:
                    st.success("✅ Likely Real News")
                else:
                    st.info(f"Model prediction: {prediction}")

                # Optional: Show prediction probability if supported
                if hasattr(model, "predict_proba"):
                    probs = model.predict_proba(features)[0]
                    confidence = max(probs) * 100
                    st.write(f"**Model Confidence:** {confidence:.2f}%")

                st.caption(
                    "This is a statistical model prediction based on text patterns, not a live fact-checker. "
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

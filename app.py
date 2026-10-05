import streamlit as st
import joblib

st.set_page_config(
    page_title="Email Spam Detection",
    page_icon="📧",
    layout="centered"
)

model = joblib.load("dataset/spam_model.pkl")
tfidf = joblib.load("dataset/tfidf_vectorizer.pkl")

st.title("Email Spam Detection")

st.write(
    "Enter an email or message below and our Machine Learning model "
    "will predict whether it is Spam or Not Spam."
)

message = st.text_area(
    "Enter your email/message:",
    placeholder="Example: Congratulations! You have won a free prize..."
)

if st.button("🔍 Predict"):

    if message.strip() == "":
        st.warning("Please enter a message first.")

    else:
        message_tfidf = tfidf.transform([message])

        prediction = model.predict(message_tfidf)
        probability = model.predict_proba(message_tfidf)

        spam_probability = probability[0][1] * 100

        if prediction[0] == "spam":
            st.error("This message is SPAM")
            st.write(f"Spam Probability: {spam_probability:.2f}%")
        else:
            st.success("This message is NOT SPAM")
            st.write(f"Spam Probability: {spam_probability:.2f}%")
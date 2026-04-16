
import streamlit as st  # GUI framework
import pandas as pd  # data handling
import matplotlib.pyplot as plt  # graphs
from sklearn.feature_extraction.text import TfidfVectorizer  # text processing
from sklearn.linear_model import LogisticRegression  # ML model

st.set_page_config(page_title="AI Phishing Detector", layout="centered")

st.title("🔐 AI Phishing Detector")
st.markdown("### Detect suspicious messages using AI")

st.write("Enter a message below to check if it is phishing or legitimate.")

data = {
    "text": [
        "Your account has been suspended. Click here immediately.",
        "Win a free iPhone now!!!",
        "Meeting at 5 PM tomorrow.",
        "Verify your bank account urgently.",
        "Let's study for the exam together."
    ],
    "label": [1, 1, 0, 1, 0]
}

df = pd.DataFrame(data)


vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df["text"])
y = df["label"]



model = LogisticRegression()
model.fit(X, y)

suspicious_words = ["urgent", "click", "verify", "password", "account", "win"]


def highlight_words(text):
    words = text.split()
    highlighted = []

    for word in words:
        clean = word.lower().strip("!.,")
        if clean in suspicious_words:
            highlighted.append(f"🔴 **{word}**")
        else:
            highlighted.append(word)

    return " ".join(highlighted)



user_input = st.text_area("📩 Enter Message Here:")

if st.button("🔍 Analyze Message"):

    if user_input.strip() == "":
        st.warning("⚠️ Please enter a message first.")

    else:
        # Convert message to vector
        vec = vectorizer.transform([user_input])

        # Get probabilities
        prob = model.predict_proba(vec)[0]

        legit_prob = prob[0]
        phishing_prob = prob[1]

        # Prediction logic
        if phishing_prob > 0.5:
            st.error("🚨 Phishing Detected!")
        else:
            st.success("✅ Legitimate Message")

        # Confidence
        st.write(f"📊 **Confidence (Phishing): {phishing_prob:.2f}**")

        # Highlighted message
        st.markdown("### 🔍 Highlighted Message")
        st.markdown(highlight_words(user_input))

       

        fig, ax = plt.subplots()
        ax.bar(["Legitimate", "Phishing"], [legit_prob, phishing_prob])
        ax.set_title("AI Confidence")

        st.pyplot(fig)


st.markdown("---")
st.markdown("⚠️ Educational Use Only")
st.markdown("Built for AISC Session 🚀")

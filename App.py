import streamlit as st
import pickle
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
import speech_recognition as sr
import io

# Load the pickled tokenizer and model
tokenizer = pickle.load(open('tokenizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

st.title("🌟 Text Or Audio Sentiment Analysis 🌟")

def predict_sentiment(text):
    tokenized = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(tokenized, maxlen=200)
    prediction = model.predict(padded)
    sentiment = "Positive" if prediction[0] > 0.5 else "Negative"
    confidence = float(prediction[0]) if sentiment == "Positive" else 1 - float(prediction[0])
    return sentiment, confidence

# Input mode
input_mode = st.radio("Choose input method:", ["⌨️ Type Text", "🎙️ Speak"])

user_input = ""
analyze_button = False

# ── Text mode ──────────────────────────────────────────────
if input_mode == "⌨️ Type Text":
    user_input = st.text_area("Enter your text:", "")
    analyze_button = st.button("🔍 Analyze Sentiment")

# ── Audio mode (uses built-in Streamlit audio_input) ───────
else:
    st.info("🎙️ Click the mic button below to record your voice")

    # Built-in Streamlit mic recorder — NO ffmpeg needed!
    audio_bytes = st.audio_input("Record your message")

    if audio_bytes is not None:
        st.audio(audio_bytes, format="audio/wav")

        recognizer = sr.Recognizer()

        with sr.AudioFile(io.BytesIO(audio_bytes.read())) as source:
            audio_data = recognizer.record(source)
            try:
                user_input = recognizer.recognize_google(audio_data)
                st.success(f"📝 Transcribed: **{user_input}**")
                analyze_button = True
            except sr.UnknownValueError:
                st.error("❌ Could not understand. Please speak clearly and try again.")
            except sr.RequestError:
                st.error("❌ Internet connection needed for speech recognition.")

# ── Sentiment Result ───────────────────────────────────────
if analyze_button and user_input.strip():
    sentiment, confidence = predict_sentiment(user_input)

    if sentiment == "Positive":
        emoji = "😊"
        s_color = "color:green; font-size:26px;"
        c_color = "color:green; font-size:20px;"
    else:
        emoji = "😞"
        s_color = "color:red; font-size:26px;"
        c_color = "color:red; font-size:20px;"

    st.markdown(f"<p style='{s_color}'>{emoji} <b>Predicted Sentiment:</b> {sentiment}</p>", unsafe_allow_html=True)
    st.markdown(f"<p style='{c_color}'>📊 <b>Confidence:</b> {confidence:.2%}</p>", unsafe_allow_html=True)
<div align="center">

# 🎭 Sentiment Analysis using LSTM

### 🎙️ Voice & ⌨️ Text Based Sentiment Analysis using Deep Learning

<p align="center">

![Python](https://img.shields.io/badge/Python-3.x-blue?style=for-the-badge&logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow)
![Keras](https://img.shields.io/badge/Keras-D00000?style=for-the-badge&logo=keras)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit)
![NLTK](https://img.shields.io/badge/NLTK-154F5B?style=for-the-badge)

</p>

### 😊 Predict Human Sentiment from Text and Voice using LSTM

A Deep Learning web application built with **Streamlit** that predicts the sentiment of a user's text or voice input using an **LSTM (Long Short-Term Memory)** neural network.

⭐ **If you like this project, please give it a Star!**

</div>

---

# 📖 Project Overview

Sentiment Analysis is one of the most popular Natural Language Processing (NLP) applications.

This project classifies user reviews or sentences into **Positive** or **Negative** sentiment using a trained **LSTM Deep Learning model**.

Users can either:

- ⌨️ Type a sentence
- 🎤 Speak using their microphone

The application automatically predicts the sentiment in real time.

---

# ✨ Features

- 😊 Positive / Negative Sentiment Prediction
- 🎤 Voice Input Support
- ⌨️ Text Input Support
- 🧠 LSTM Deep Learning Model
- 📊 Interactive Streamlit UI
- ⚡ Real-Time Prediction
- 📝 IMDB Movie Review Dataset
- 💾 Saved Model (.pkl)
- 🚀 Fast Prediction

---

# 🛠️ Technologies Used

| Technology | Purpose |
|------------|---------|
| Python | Programming Language |
| TensorFlow | Deep Learning |
| Keras | LSTM Model |
| Streamlit | Web Application |
| Pandas | Data Processing |
| NumPy | Numerical Computing |
| NLTK | Text Preprocessing |
| Pickle | Model Serialization |
| SpeechRecognition | Voice Input |

---

# 📂 Project Structure

```text
SENTIMENT_ANALYSIS/
│
├── App.py
├── IMDB_data.csv
├── SENTIMENT ANALYSIS.ipynb
├── model.pkl
├── tokenizer.pkl
├── README.md
│
└── images/
    ├── home.png
    ├── typing_prediction.png
    ├── voice_prediction.png
    ├── result_positive.png
    └── model_architecture.png
```

---

# 🧠 Deep Learning Model

The project uses an **LSTM (Long Short-Term Memory)** neural network.

### Model Pipeline

```text
Input Text
     │
     ▼
Text Cleaning
     │
     ▼
Tokenization
     │
     ▼
Padding Sequences
     │
     ▼
Embedding Layer
     │
     ▼
LSTM Layer
     │
     ▼
Dense Layer
     │
     ▼
Sigmoid Activation
     │
     ▼
Positive / Negative
```

---

# 📊 Dataset

Dataset Used:

🎬 **IMDB Movie Reviews Dataset**

Contains thousands of labeled movie reviews.

Each review is classified as:

- 😊 Positive
- 😞 Negative

---

# 🎯 Input Methods

The application supports two different input methods.

### ⌨️ Typing

Users can type any sentence or review.

Example:

```
This movie was absolutely amazing.
```

---

### 🎤 Voice

Users can click the microphone button and speak naturally.

Example:

```
I really enjoyed this movie.
```

The voice is converted into text and then analyzed by the LSTM model.

---

# 🔄 Project Workflow

```text
User Input
(Text / Voice)
      │
      ▼
Speech to Text (Voice)
      │
      ▼
Text Cleaning
      │
      ▼
Tokenization
      │
      ▼
Padding
      │
      ▼
LSTM Prediction
      │
      ▼
Positive / Negative
      │
      ▼
Display Result
```

---

# 📷 Application Screenshots

## 🏠 Home Page

<p align="center">
<img src="images/home.png" width="900">
</p>

The home page provides a clean and interactive interface where users can enter text manually or choose voice input for sentiment prediction.

---

## ⌨️ Text Sentiment Prediction

<p align="center">
<img src="images/typing_prediction.png" width="900">
</p>

Users can type any sentence or review into the text box. The LSTM model processes the input and instantly predicts whether the sentiment is positive or negative.

---

## 🎤 Voice Sentiment Prediction

<p align="center">
<img src="images/voice_prediction.png" width="900">
</p>

Users can record their voice using the microphone. The spoken words are converted into text using speech recognition before being analyzed by the LSTM model.

---

## 😊 Prediction Result

<p align="center">
<img src="images/result_positive.png" width="900">
</p>

The application displays the predicted sentiment along with an intuitive and user-friendly result screen.

---

## 🧠 LSTM Model Architecture

<p align="center">
<img src="images/model_architecture.png" width="900">
</p>

The figure illustrates the architecture of the LSTM network used for sentiment classification.

---

# 🚀 Installation

### Clone Repository

```bash
git clone https://github.com/yourusername/SENTIMENT-ANALYSIS-LSTM.git
```

---

### Go to Project Directory

```bash
cd SENTIMENT-ANALYSIS-LSTM
```

---

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

### Run Application

```bash
streamlit run App.py
```

---

# 📌 Future Improvements

- 🌍 Multi-language Support
- 😀 Emotion Detection
- 📈 Confidence Score
- ☁️ Cloud Deployment
- 📱 Mobile Friendly UI
- 🤖 Transformer Models (BERT)

---

# 🤝 Contributing

Contributions are welcome.

1. Fork this repository

2. Create a branch

```bash
git checkout -b feature-name
```

3. Commit changes

```bash
git commit -m "Added new feature"
```

4. Push

```bash
git push origin feature-name
```

5. Create a Pull Request

---

# 👨‍💻 Author

## Rohit Dutta

🎓 Artificial Intelligence & Machine Learning Student

📧 **Email:** ronidutta854@gmail.com

🐙 **GitHub:** https://github.com/rohit-2025-ops

---

<div align="center">

## ❤️ Thank You for Visiting ❤️

### ⭐ If you found this project helpful, please give it a Star ⭐

Made with ❤️ by **Rohit Dutta**

</div>

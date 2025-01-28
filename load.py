from flask import Flask, request, render_template, jsonify
import tensorflow as tf
import numpy as np
import joblib
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import nltk

nltk.download('stopwords')

app = Flask(__name__)

# Load TensorFlow Lite model
interpreter = tf.lite.Interpreter(model_path="logistic_regression_model.tflite")
interpreter.allocate_tensors()

# Load the TF-IDF vectorizer
vectorizer = joblib.load("tfidf_vectorizer.joblib")

# Preprocessing function
def preprocess_text(text):
    text = text.lower().split()
    stop_words = set(stopwords.words("english"))
    stemmer = PorterStemmer()
    tokens = [stemmer.stem(word) for word in text if word not in stop_words]
    return " ".join(tokens)

@app.route("/", methods=["GET", "POST"])
def index():
    prediction_result = None
    conclusion_result = None  # Ensure the variable is always defined

    if request.method == "POST":
        user_text = request.form["news_text"]
        processed_text = preprocess_text(user_text)

        # Convert text to TF-IDF features
        input_features = vectorizer.transform([processed_text]).toarray()

        # Set up the TensorFlow Lite interpreter
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        interpreter.set_tensor(input_details[0]['index'], input_features.astype(np.float32))
        interpreter.invoke()
        output = interpreter.get_tensor(output_details[0]['index'])

        # Compute the prediction result
        prediction_result = f"{output[0][0] * 100:.2f}% chance of being fake news"
        if output[0][0] < 0.2:
            conclusion_result = "The news is legitimate"
        else:
            conclusion_result = "This is fake news"

    return render_template("index.html", prediction=prediction_result, conclusion=conclusion_result)


if __name__ == "__main__":
    app.run(debug=True)

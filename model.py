import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import tensorflow as tf
import numpy as np
import joblib

# Download necessary NLTK resources
nltk.download('stopwords')

# Load the datasets
fake_df = pd.read_csv('fake.csv')
true_df = pd.read_csv('true.csv')

# Label the datasets
fake_df['label'] = 1  # Fake news is labeled as 1
true_df['label'] = 0  # Real news is labeled as 0


# Combine the two datasets
df = pd.concat([fake_df, true_df], ignore_index=True)

# Sample text preprocessing function
def preprocess_text(text):
    # Tokenization & Lowercasing
    tokens = text.lower().split()

    # Remove stopwords and apply stemming
    stop_words = set(stopwords.words('english'))
    stemmer = PorterStemmer()

    tokens = [stemmer.stem(word) for word in tokens if word not in stop_words]

    return " ".join(tokens)

# Apply preprocessing to the text column
df['processed_text'] = df['text'].apply(preprocess_text)

# Extract features (TF-IDF) and labels
X = df['processed_text']
y = df['label']

# Convert text to numerical format using TF-IDF
vectorizer = TfidfVectorizer(max_features=5000)
X_tfidf = vectorizer.fit_transform(X)

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.2, random_state=42)

# Train a Logistic Regression model
logreg_model = LogisticRegression(max_iter=200)
logreg_model.fit(X_train, y_train)

# Evaluate the model
y_pred = logreg_model.predict(X_test)
print(classification_report(y_test, y_pred))

# Save the trained Logistic Regression model
joblib.dump(logreg_model, 'logistic_regression_model.joblib')
joblib.dump(vectorizer, 'tfidf_vectorizer.joblib')

# Convert the Logistic Regression model to TensorFlow
weights = logreg_model.coef_.flatten()
bias = logreg_model.intercept_[0]

# Create a simple TensorFlow model that mimics Logistic Regression
tf_model = tf.keras.Sequential([
    tf.keras.layers.Dense(1, input_dim=X_train.shape[1], activation='sigmoid', use_bias=True)
])

# Set the weights and bias manually to match the Logistic Regression model
tf_model.layers[0].set_weights([weights.reshape(-1, 1), np.array([bias])])

# Print TensorFlow model summary
tf_model.summary()

# Convert the TensorFlow model to TensorFlow Lite format
converter = tf.lite.TFLiteConverter.from_keras_model(tf_model)
tflite_model = converter.convert()

# Save the TensorFlow Lite model to a file
with open('logistic_regression_model.tflite', 'wb') as f:
    f.write(tflite_model)

print("Model and vectorizer saved successfully.")
print("TensorFlow Lite model saved as 'logistic_regression_model.tflite'.")

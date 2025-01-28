import tensorflow as tf
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Create a simple Logistic Regression model
logreg_model = LogisticRegression(max_iter=200)

# Assume X_train_tfidf and y_train are your preprocessed data and labels
logreg_model.fit(X_train_tfidf, y_train)

# Get the model's weights
weights = logreg_model.coef_.flatten()
bias = logreg_model.intercept_[0]

# Create a TensorFlow model that mimics the Logistic Regression
tf_model = Sequential([
    Dense(1, input_dim=X_train_tfidf.shape[1], activation='sigmoid', use_bias=True)
])

# Set the weights and bias manually to match the Logistic Regression model
tf_model.layers[0].set_weights([weights.reshape(-1, 1), np.array([bias])])

# Summary of the TensorFlow model
tf_model.summary()


# Convert the TensorFlow model to TensorFlow Lite format
converter = tf.lite.TFLiteConverter.from_keras_model(tf_model)
tflite_model = converter.convert()

# Save the TensorFlow Lite model to a file
with open('logistic_regression_model.tflite', 'wb') as f:
    f.write(tflite_model)

print("Model successfully converted to TensorFlow Lite format.")

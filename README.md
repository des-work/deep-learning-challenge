# Machine Learning Project: Charity Success Prediction

## Overview
This project develops a neural network model to predict the success of charity organizations based on their application data.

## Data Preprocessing
Significant features include application type and classification. Infrequent categories within these features are labeled as "Other".

## Neural Network Model
The model is a sequential neural network built with TensorFlow and Keras, consisting of two hidden layers with ReLU activation and an output layer with sigmoid activation.

## Training
The network is trained for 100 epochs with a binary cross-entropy loss function and Adam optimizer.

## Evaluation
The model achieved ~72% accuracy on the test data, indicating a strong base model with room for further optimization.

## Code Snippet
```python
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Define the model
model = Sequential([
    Dense(128, activation='relu', input_shape=(num_features,)),
    Dense(60, activation='relu'),
    Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=100, validation_data=(X_test, y_test))

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Accuracy: {accuracy * 100:.2f}%")

# Machine Learning Project

In this project, I've demonstrated my skills in handling real-world data, preprocessing, building a neural network model, and evaluating its performance. Let's dive into the details.

## Overview

The goal of this project is to predict whether a charity organization will be successful based on various features. The dataset includes information such as application type, affiliation, classification, and more.

## Project Structure

- **Data Understanding and Cleaning**
  - **Application Types:** I identified and replaced infrequent application types with "Other" to simplify the dataset.
  - **Classifications:** I grouped less frequent classifications into "Other" for better model generalization.
  - **Categorical Encoding:** Converted categorical data into a numeric format using one-hot encoding.

```python
# Data Cleaning - Application Types
cutoff_value = 500
application_types_to_replace = unique_values[unique_values < cutoff_value].index.tolist()
for app in application_types_to_replace:
    application_df['APPLICATION_TYPE'] = application_df['APPLICATION_TYPE'].replace(app, "Other")

# Data Cleaning - Classifications
cutoff_value = 15
classification_counts = application_df['CLASSIFICATION'].value_counts()
classifications_to_replace = classification_counts[classification_counts < cutoff_value].index.tolist()
for cls in classifications_to_replace:
    application_df['CLASSIFICATION'] = application_df['CLASSIFICATION'].replace(cls, "Other")

# Convert Categorical Data
application_df_encoded = pd.get_dummies(application_df)
```

- **Model Building and Training**
  - **Neural Network:** I constructed a neural network using TensorFlow's Keras API.
  - **Structure:** The model consists of three layers, including two hidden layers with ReLU activation.
  - **Training:** The model was trained on preprocessed data for 100 epochs.

```python
# Define the model
nn = tf.keras.models.Sequential()
nn.add(tf.keras.layers.Dense(units=128, input_dim=INPUT_FEATURES, activation='relu'))
nn.add(tf.keras.layers.Dense(units=60, activation='relu'))
nn.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))

# Compile the model
nn.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
nn.fit(X_train_scaled, y_train, epochs=100, validation_data=(X_test_scaled, y_test))
```

- **Model Evaluation and Export**
  - **Evaluation:** The model achieved an accuracy of approximately 72% on the test set.
  - **Export:** The trained model was saved in HDF5 format.

```python
# Evaluate the model
model_loss, model_accuracy = nn.evaluate(X_test_scaled, y_test, verbose=2)
print(f"Loss: {model_loss}, Accuracy: {model_accuracy}")

# Export the model
nn.save("charity_model.h5")
```

## Conclusion

This project showcases my end-to-end machine learning skills, from data preprocessing to building and evaluating a neural network model. The model's performance provides valuable insights into predicting charity success. Feel free to explore the code and model for a deeper understanding.

Thank you!

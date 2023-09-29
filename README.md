# deep-learning-challenge
 Binary classifier that can predict whether applicants will be successful if funded by Alphabet Soup.


1. Data Understanding
The dataset contained diverse information about different organizations, including application types, affiliations, classifications, and more.

2. Exploring the Data
I examined the data to understand the distribution of application types and classifications, identifying patterns and insights.


# Display the first few rows of the dataset
application_df.head()
3. Data Cleaning
Observing that some categories had limited occurrences, I decided to simplify the model by grouping them under an 'Other' category. For instance, application types with fewer than 500 occurrences were combined into 'Other.'


# Choose a cutoff value and create a list of application types to be replaced
cutoff_value = 500
application_types_to_replace = unique_values[unique_values < cutoff_value].index.tolist()

# Replace in dataframe
for app in application_types_to_replace:
    application_df['APPLICATION_TYPE'] = application_df['APPLICATION_TYPE'].replace(app, "Other")
4. Handling Classification Data
Similar to application types, I grouped classifications with less than 15 occurrences into an 'Other' category for simplification.


# Look at CLASSIFICATION value counts for binning
classification_counts = application_df['CLASSIFICATION'].value_counts()

# Choose a cutoff value and create a list of classifications to be replaced
cutoff_value = 15
classifications_to_replace = classification_counts[classification_counts < cutoff_value].index.tolist()

# Replace in dataframe
for cls in classifications_to_replace:
    application_df['CLASSIFICATION'] = application_df['CLASSIFICATION'].replace(cls, "Other")
5. Converting Categorical Data
To facilitate machine learning, I converted categorical variables into a numerical format using one-hot encoding.


# Convert categorical data to numeric with `pd.get_dummies`
application_df_encoded = pd.get_dummies(application_df)
6. Data Splitting
The data was divided into features and target variables, followed by a split into training and testing datasets.


# Split our preprocessed data into our features and target arrays
X = application_df_encoded.drop("IS_SUCCESSFUL", axis=1)
y = application_df_encoded["IS_SUCCESSFUL"]

# Split the preprocessed data into a training and testing dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=78, stratify=y)
7. Building the Neural Network Model
A neural network model with three layers—input, hidden, and output—was constructed. The number of nodes in each layer was determined based on data characteristics.


# Define the model - deep neural net
nn = tf.keras.models.Sequential()

# First hidden layer
nn.add(tf.keras.layers.Dense(units=128, input_dim=len(X_train.columns), activation='relu'))

# Second hidden layer
nn.add(tf.keras.layers.Dense(units=60, activation='relu'))

# Output layer
nn.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))

# Check the structure of the model
nn.summary()
8. Model Training
The model underwent training using the training dataset, optimizing parameters across multiple epochs.

# Compile the model
nn.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
model.fit(X_train_scaled, y_train, epochs=100)
9. Evaluating Model Performance
Post-training, the model's performance was assessed using the testing dataset, with metrics such as accuracy being analyzed.


# Evaluate the model using the test data
model_loss, model_accuracy = nn.evaluate(X_test_scaled, y_test, verbose=2)
print(f"Loss: {model_loss}, Accuracy: {model_accuracy}")
10. Model Preservation
Ultimately, the trained model was saved for potential future applications.


# Export our model to HDF5 file
nn.save("trained_charity.h5")

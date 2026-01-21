# Import necessary libraries
try:
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.metrics import accuracy_score, classification_report
except ModuleNotFoundError as e:
    print(f"Required module not found: {e.name}. Please install it using 'pip install {e.name}' and try again.")
    exit()

# Load the dataset
try:
    data = pd.read_csv('/content/spam.csv', encoding='latin-1')
    print("Dataset loaded successfully.")
except FileNotFoundError:
    print("Dataset file 'spam.csv' not found. Please ensure the file is in the correct location.")
    exit()
except Exception as e:
    print(f"Error loading dataset: {e}")
    exit()

# Inspect the dataset
print("Preview of the dataset:")
print(data.head())

# Preprocess the data
data = data[['v1', 'v2']]
data.columns = ['label', 'message']
data['label'] = data['label'].map({'ham': 0, 'spam': 1})

# Split the data into training and testing sets
X = data['message']
y = data['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Convert text to feature vectors using CountVectorizer
vectorizer = CountVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train a Naive Bayes model
model = MultinomialNB()
model.fit(X_train_vec, y_train)

# Make predictions
y_pred = model.predict(X_test_vec)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"\nModel Accuracy: {accuracy:.2f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Take user input for custom message and predict
user_message = input("\nEnter the email content to check if it is Spam or Not: ")
user_message_vec = vectorizer.transform([user_message])
user_prediction = "Spam" if model.predict(user_message_vec)[0] == 1 else "Not Spam"
print(f"\nPrediction for the entered email: {user_prediction}")
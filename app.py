import os
import json
import random
import nltk
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Check for nltk data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# Load intents.json
file_path = os.path.abspath("C:\\Users\\Admin\\Documents\\intents.json")  # Update path if necessary
try:
    with open(file_path, "r") as file:
        intents = json.load(file)
except FileNotFoundError:
    print(f"Error: File not found at {file_path}. Please check the path.")
    exit()

# Initialize vectorizer and classifier
vectorizer = TfidfVectorizer()
clf = LogisticRegression(random_state=0, max_iter=100000)

# Extract patterns and tags
tags = []
patterns = []
for intent in intents.get('intents', []):
    if 'patterns' not in intent or 'responses' not in intent:
        continue
    for pattern in intent['patterns']:
        tags.append(intent['tag'])
        patterns.append(pattern)

# Check if data is available for training
if not patterns or not tags:
    print("Error: No patterns or tags found in the intents dataset.")
    exit()

# Transform the patterns into TF-IDF vectors
x = vectorizer.fit_transform(patterns)
y = tags

# Train the Logistic Regression classifier
clf.fit(x, y)

# Save the trained model and vectorizer
joblib.dump(clf, "clf.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")

# Chatbot function
def chatbot(input_text):
    input_text_vector = vectorizer.transform([input_text])
    predicted_tag = clf.predict(input_text_vector)[0]
    
    for intent in intents['intents']:
        if intent['tag'] == predicted_tag:
            return random.choice(intent['responses'])
    
    return "I'm sorry, I didn't understand that."

# Interactive Chat
if __name__ == "__main__":
    print("Chatbot is ready! Type 'quit' or 'exit' to end the conversation.")
    while True:
        user_input = input("You: ")
        if user_input.lower() in ['quit', 'exit']:
            print("Chatbot: Goodbye!")
            break
        print(f"Chatbot: {chatbot(user_input)}")

import json
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

with open("intents.json") as file:
    data = json.load(file)

patterns = []
tags = []

for intent in data["intents"]:
    for pattern in intent["patterns"]:
        patterns.append(pattern.lower())   # 🔥 important fix
        tags.append(intent["tag"])

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(patterns)

model = MultinomialNB()
model.fit(X, tags)

pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))

print("Model trained successfully!")
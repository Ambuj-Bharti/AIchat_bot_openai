import json
import random
import pickle
import re

# Load data
with open("intents.json") as file:
    data = json.load(file)

model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

print("🤖 AI Chatbot Started (type 'exit' to stop)\n")

while True:
    try:
        # Clean input (keep spaces)
        user = re.sub(r'[^a-zA-Z0-9 ]', '', input("You: ")).strip().lower()

        # Exit condition
        if user == "exit":
            print("Bot: Goodbye! 👋")
            break

        # Empty input check
        if user == "":
            print("Bot: Please enter a message.")
            continue

        # Convert to vector
        X_test = vectorizer.transform([user])

        # Predict intent
        probs = model.predict_proba(X_test)
        confidence = max(probs[0])
        tag = model.predict(X_test)[0]

        print(f"(Debug → Tag: {tag}, Confidence: {confidence:.2f})")

        # ✅ SAFE LOGIC
        found = False

        if confidence > 0.5:
            for intent in data["intents"]:
                if intent["tag"] == tag:
                    print("Bot:", random.choice(intent["responses"]))
                    found = True
                    break

        # ❌ fallback (no wrong answer)
        if not found:
            print("Bot: I’m not sure about that yet. Try asking something else 😊")

    except Exception as e:
        print("Error:", e)
        print("Bot: Something went wrong.")
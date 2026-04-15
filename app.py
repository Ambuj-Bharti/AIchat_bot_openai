from flask import Flask, render_template, request, jsonify
import json
import random
import pickle
import os
import re
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

app = Flask(__name__)

client = OpenAI(api_key=os.getenv("YOUR_API_KEY"))

# Load intents
with open("intents.json") as file:
    data = json.load(file)

# Load trained model
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    try:
        # Clean input
        user = re.sub(r'[^a-zA-Z0-9 ]', '', request.json.get("message", "")).strip().lower()

        if user == "":
            return jsonify({"reply": "Please enter a message."})

        # Vectorize
        X_test = vectorizer.transform([user])

        # Predict
        probs = model.predict_proba(X_test)
        confidence = max(probs[0])
        tag = model.predict(X_test)[0]

        print("User:", user, "| Tag:", tag, "| Confidence:", confidence)

        # ✅ SAFE LOGIC
        found = False

        if confidence > 0.5:
            for intent in data["intents"]:
                if intent["tag"] == tag:
                    found = True
                    return jsonify({
                        "reply": random.choice(intent["responses"])
                    })

        # ❌ fallback (no wrong answer)
        if not found:
            return jsonify({
                "reply": "I’m not sure about that yet. Try asking something else 😊"
            })

    except Exception as e:
        print("Error:", e)
        return jsonify({"reply": "Server error occurred"})


# ✅ THIS MUST BE OUTSIDE (VERY IMPORTANT)
if __name__ == "__main__":
    app.run(debug=True)
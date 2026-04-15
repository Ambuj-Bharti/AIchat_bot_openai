from openai import OpenAI
import os
from dotenv import load_dotenv

#load .env file

load_dotenv()

client = OpenAI(api_key=os.getenv("YOUR_API_KEY"))

try:
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "user", "content": "Explain what Python programming language is."}
        ]
    )

    print("OpenAI API is working!\n")
    print("AI Response:\n")
    print(response.choices[0].message.content)

except Exception as e:
    print("Error occurred:")
    print(e)
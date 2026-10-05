from dotenv import load_dotenv
from google import genai

load_dotenv()
client = genai.Client()

chat = client.chats.create(
    model="gemini-3.8-flash"
)

print("Gemini Dynamic Chatbot")
print("Type 'exit' to quit.\n")


while True:

    question = input("You: ")

    if question.lower() == "exit":
        print("Goodbye!")
        break

    try:

        response = chat.send_message(
            question
        )

        print("Gemini:", response.text)
        print()

    except Exception as e:

        print("API Error:", e)
        print("Please try again.\n")
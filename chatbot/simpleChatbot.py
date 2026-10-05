from google import genai


client=genai.Client()

response = client.models.generate_content(
    model="gemini-3.8-flash",
    contents="What is software testing? Explain in simple words."
)

print(response.text)
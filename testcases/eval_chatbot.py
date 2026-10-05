from dotenv import load_dotenv
from google import genai
from deepeval import evaluate
from deepeval.test_case import LLMTestCase
from deepeval.metrics import AnswerRelevancyMetric

load_dotenv()

# Create Gemini client
client = genai.Client()

def ask_chatbot(question):
    response = client.models.generate_content(
        model="gemini-3.6-flash",
        contents=question
    )

    return response.text


# Test input
question = "What is Playwright? Answer in 2 sentences."

# Get REAL response from Gemini chatbot
actual_response = ask_chatbot(question)

print("\nGemini response:")
print(actual_response)

# Create DeepEval test case
test_case = LLMTestCase(
    input=question,
    actual_output=actual_response
)

# Answer Relevancy metric
metric = AnswerRelevancyMetric(
    threshold=0.7,
    include_reason=True
)

# Run evaluation
evaluate(
    test_cases=[test_case],
    metrics=[metric]
)
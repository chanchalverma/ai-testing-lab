from dotenv import load_dotenv
from google import genai

from deepeval import evaluate
from deepeval.test_case import Turn, ConversationalTestCase
from deepeval.metrics import (
    TurnRelevancyMetric,
    KnowledgeRetentionMetric,
    ConversationCompletenessMetric,
)

# --------------------------------------------------
# 1. Load environment variables
# --------------------------------------------------

load_dotenv()


# --------------------------------------------------
# 2. Create Gemini client and chat
# --------------------------------------------------

client = genai.Client()

chat = client.chats.create(
    model="gemini-3.6-flash"
)


# --------------------------------------------------
# 3. Function to get REAL Gemini response
# --------------------------------------------------

def ask_gemini(question):

    response = chat.send_message(question)

    return response.text


# --------------------------------------------------
# 4. Conversation
# --------------------------------------------------

# For trial purpose, donot query more as limited quota (2-3)
questions = [
    "What is Playwright? Answer in 2 sentences.",
    # "I mainly use JavaScript. Can I use Playwright with JavaScript?",
    # "Show me a simple Playwright JavaScript example.",
    # "How is Playwright different from Selenium?"
]


# --------------------------------------------------
# 5. Capture actual conversation
# --------------------------------------------------

turns = []

print("\n==========================================")
print("       GEMINI CONVERSATIONAL TEST")
print("==========================================\n")


for question in questions:

    print("User:")
    print(question)

    # Add actual user turn
    turns.append(
        Turn(
            role="user",
            content=question
        )
    )

    # Get REAL Gemini response
    answer = ask_gemini(question)

    print("\nGemini:")
    print(answer)

    # Add actual Gemini response
    turns.append(
        Turn(
            role="assistant",
            content=answer
        )
    )

    print("\n------------------------------------------\n")


# --------------------------------------------------
# 6. Create ConversationalTestCase
# --------------------------------------------------

conversation_test = ConversationalTestCase(
    turns=turns,

    scenario=(
        "A user is learning Playwright and asks follow-up "
        "questions about JavaScript, examples, and Selenium."
    ),

    expected_outcome=(
        "The chatbot should provide relevant and accurate answers, "
        "remember the user's preference for JavaScript, and "
        "maintain context throughout the conversation."
    )
)


# --------------------------------------------------
# 7. Define conversational metrics
# --------------------------------------------------

turn_relevancy = TurnRelevancyMetric(
    threshold=0.7
)

knowledge_retention = KnowledgeRetentionMetric(
    threshold=0.7
)

conversation_completeness = ConversationCompletenessMetric(
    threshold=0.7
)


# --------------------------------------------------
# 8. Evaluate conversation
# --------------------------------------------------

print("\n==========================================")
print("          DEEPEVAL EVALUATION")
print("==========================================\n")


evaluate(
    test_cases=[conversation_test],
    metrics=[
        turn_relevancy,
        knowledge_retention,
        conversation_completeness
    ]
)
from openai import OpenAI
from dotenv import load_dotenv
import os
from utils import retrieve_top_k_chunks_sentencetransformers
from utils import retrieve_top_k_chunks_openai #not used

PATH_TO_PROMPTS = "/Users/sydneydu/Projects/intro_ml_rag_assistant/prompts"
load_dotenv()
SYSTEM_PROMPT = open(f"{PATH_TO_PROMPTS}/system.txt", "r").read().strip()
    
model = OpenAI(api_key = os.getenv("OPENAI_API_KEY"))

def call_llm(query: str, context = []):
    context = "\n".join(context)
    response = model.responses.create(
    model="gpt-4.1",
    input=[
        {
            "role": "system",
            "content": SYSTEM_PROMPT
        },
        {
            "role": "user",
            "content": f"Question: {query}?\n Context from the course textbook: {context}"
        }
        ]
    )
    return response
def ask(query: str):
    """Returns the string answering the question"""
    context = retrieve_top_k_chunks_sentencetransformers(query, k = 20)
    response = call_llm(query, context)
    return response.output[0].content[0].text

if __name__ == "__main__":
    query = "What are the stopping conditions for the gradient descent algorithm?"
    print(ask(query))
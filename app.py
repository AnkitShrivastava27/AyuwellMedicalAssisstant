import os
import google.generativeai as genai
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configure the Gemini API key
API_KEY = os.getenv("GOOGLE_API_KEY")
if not API_KEY:
    raise RuntimeError("GOOGLE_API_KEY environment variable not set.")
genai.configure(api_key=API_KEY)

# Define the system prompt (medical assistant role with word limit)
SYSTEM_PROMPT = (
    "You are a helpful medical assistant.\n"
    "- You guide patients/users for **general medical queries**.\n"
    "- Politely decline any **non-medical requests**.\n"
    "- You may reply politely to simple greetings like 'hi' or 'hello'.\n"
    "- Always remind users that in **severe or emergency cases**, they should consult a doctor immediately.\n"
    "- Your answers must always be **concise** and must not exceed **30 words**."
)

# Initialize the FastAPI app
app = FastAPI(
    title="Gemini Medical Assistant API",
    description="An API that uses Gemini to act as a medical assistant for general medical queries, "
                "with short answers (max 40 words).",
    version="1.0.0",
)

# Initialize the Gemini Pro model
model = genai.GenerativeModel("gemini-1.5-flash")

# Define the request body model using Pydantic
class PromptRequest(BaseModel):
    prompt: str

# Define the API endpoint
@app.post("/generate")
async def generate_response(request: PromptRequest):
    """
    Accepts a user's prompt and returns a response from the Gemini model.
    """
    if not request.prompt.strip():
        raise HTTPException(status_code=400, detail="Prompt cannot be empty.")

    try:
        # Send the system prompt + user prompt to Gemini
        response = await model.generate_content_async(
            [SYSTEM_PROMPT, request.prompt]
        )

        return {"response": response.text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

# Define a root endpoint for health check
@app.get("/")
def read_root():
    return {"status": "Medical Assistant API is running"}

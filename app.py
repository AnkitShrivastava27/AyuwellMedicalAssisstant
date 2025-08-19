import os
import google.generativeai as genai
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
import uvicorn


load_dotenv()


API_KEY = os.getenv("GOOGLE_API_KEY")
if not API_KEY:
    raise RuntimeError("GOOGLE_API_KEY environment variable not set.")
genai.configure(api_key=API_KEY)


SYSTEM_PROMPT = (
    "You are a helpful medical assistant.\n"
    "- You guide patients/users for **general medical queries**.\n"
    "- Politely decline any **non-medical requests**.\n"
    "- You may reply politely to simple greetings like 'hi' or 'hello'.\n"
    "- Always remind users that in **severe or emergency cases**, they should consult a doctor immediately.\n"
    "- Your answers must always be **concise** and must not exceed **40 words**."
)

app = FastAPI(
    title="Gemini Medical Assistant API",
    description="An API that uses Gemini to act as a medical assistant for general medical queries, "
                "with short answers (max 40 words).",
    version="1.0.0",
)


model = genai.GenerativeModel("gemini-1.5-flash")


class PromptRequest(BaseModel):
    prompt: str


@app.post("/generate")
async def generate_response(request: PromptRequest):
    """
    Accepts a user's prompt and returns a response from the Gemini model.
    """
    if not request.prompt.strip():
        raise HTTPException(status_code=400, detail="Prompt cannot be empty.")

    try:
    
        response = await model.generate_content_async(
            [SYSTEM_PROMPT, request.prompt]
        )
        return {"response": response.text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")


@app.get("/")
def read_root():
    return {"status": "Medical Assistant API is running"}


if __name__ == "__main__":
   
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)

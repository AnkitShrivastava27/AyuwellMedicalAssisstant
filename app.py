# import os
# import google.generativeai as genai
# from fastapi import FastAPI, HTTPException
# from fastapi.middleware.cors import CORSMiddleware
# from pydantic import BaseModel
# from dotenv import load_dotenv
# import uvicorn

# load_dotenv()

# API_KEY = os.getenv("GOOGLE_API_KEY")
# if not API_KEY:
#     raise RuntimeError("GOOGLE_API_KEY environment variable not set.")
# genai.configure(api_key=API_KEY)

# SYSTEM_PROMPT = (
#     "You are a helpful medical assistant.\n"
#     "- You guide patients/users for **general medical queries**.\n"
#     "- Politely decline any **non-medical requests**.\n"
#     "- You may reply politely to simple greetings like 'hi' or 'hello'.\n"
#     "- Always remind users that in **severe or emergency cases**, they should consult a doctor immediately.\n"
#     "- Your answers must always be **concise** and must not exceed **40 words**."
# )

# app = FastAPI(
#     title="Gemini Medical Assistant API",
#     description="An API that uses Gemini to act as a medical assistant for general medical queries, "
#                 "with short answers (max 40 words).",
#     version="1.0.0",
# )

# # ✅ Add CORS middleware
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],  # allow all origins
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# model = genai.GenerativeModel("models/text-bison-001")


# class PromptRequest(BaseModel):
#     prompt: str


# @app.post("/generate")
# async def generate_response(request: PromptRequest):
#     """
#     Accepts a user's prompt and returns a response from the Gemini model.
#     """
#     if not request.prompt.strip():
#         raise HTTPException(status_code=400, detail="Prompt cannot be empty.")

#     try:
#         response = await model.generate_content_async(
#             [SYSTEM_PROMPT, request.prompt]
#         )
#         return {"response": response.text}
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")


# @app.get("/")
# def read_root():
#     return {"status": "Medical Assistant API is running"}


# if __name__ == "__main__":
#     uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
from fastapi import FastAPI
from pydantic import BaseModel
from huggingface_hub import InferenceClient
import os
import re
from fastapi.middleware.cors import CORSMiddleware
from typing import List

# -----------------------------
# FastAPI setup
# -----------------------------
app = FastAPI(title="Medical Chatbot API", version="1.0")

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # allow all origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------
# Hugging Face setup
# -----------------------------
HF_API_KEY = os.getenv("HUGGINGFACEHUB_API_TOKEN")
if not HF_API_KEY:
    raise RuntimeError("Set HUGGINGFACEHUB_API_TOKEN environment variable")
client = InferenceClient(api_key=HF_API_KEY)

MODEL_NAME = "HuggingFaceH4/zephyr-7b-beta"

# -----------------------------
# Request model
# -----------------------------
class ChatRequest(BaseModel):
    message: str
    history: List[str] = []

# -----------------------------
# System prompt
# -----------------------------
SYSTEM_PROMPT = """
You are a general medical information chatbot designed to educate, guide, reduce confusion, and encourage professional medical care.

Your role is to:
- Provide clear, evidence-based general health information
- Explain symptoms, conditions, and health concepts in simple language
- Offer lifestyle, diet, wellness, and preventive health guidance
- Help users understand possible causes and when medical attention may be required
- Reduce anxiety by giving structured, factual explanations

Strict rules:
- Do NOT diagnose diseases
- Do NOT prescribe medicines, treatments, or dosages
- Do NOT claim certainty about any medical condition
- Always encourage consulting a qualified healthcare professional when symptoms are serious, persistent, or unclear

Response style:
- Keep responses concise (~200 tokens)
- Use bullet points where helpful
- Maintain a calm, supportive, and non-alarming tone

Safety & disclaimer:
- Include gentle reminders that information is educational, not medical advice
- If a condition could be serious, clearly suggest seeking professional medical care
"""

# -----------------------------
# Chat endpoint
# -----------------------------
@app.post("/chat")
async def medical_chatbot(request: ChatRequest):
    conversation = "<|system|>\n" + SYSTEM_PROMPT + "\n"

    for msg in request.history:
        conversation += f"<|user|>\n{msg}\n"

    conversation += f"<|user|>\n{request.message}\n<|assistant|>\n"

    try:
        response = client.text_generation(
            model=MODEL_NAME,
            prompt=conversation,
            max_new_tokens=200,  # approx token limit
            temperature=0.5,
        )

        reply = response.generated_text
        # Remove internal tags if any
        reply = re.sub(r"<think>.*?</think>", "", reply, flags=re.DOTALL).strip()

        return {"reply": reply}

    except Exception as e:
        return {"error": str(e)}

# -----------------------------
# Health check
# -----------------------------
@app.get("/")
def root():
    return {"message": "Medical Chatbot API is running!"}

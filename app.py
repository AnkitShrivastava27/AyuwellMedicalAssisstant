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

app = FastAPI()

# -----------------------------
# CORS
# -----------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------
# Hugging Face setup
# -----------------------------
api_key = os.getenv("HUGGINGFACEHUB_API_TOKEN")
if not api_key:
    raise RuntimeError("Set HUGGINGFACEHUB_API_TOKEN")

client = InferenceClient(api_key=api_key)
MODEL_NAME = "HuggingFaceH4/zephyr-7b-beta"

# -----------------------------
# Request schema (PROMPT ONLY)
# -----------------------------
class ChatRequest(BaseModel):
    prompt: str

# -----------------------------
# System Prompt
# -----------------------------
SYSTEM_PROMPT = """
You are a general medical information chatbot.

Rules:
- Provide educational health information only
- Do NOT diagnose diseases
- Do NOT prescribe medicines or dosages
- Avoid certainty; use cautious language
- Encourage consulting a qualified doctor when needed

Style:
- Calm, supportive, concise
- Avoid long explanations
"""

# -----------------------------
# Helper: prompt → Zephyr format
# -----------------------------
def build_zephyr_prompt(user_prompt: str) -> str:
    return f"""
<System>
{SYSTEM_PROMPT}
</System>

<User>
{user_prompt}
</User>

<Assistant>
"""

# -----------------------------
# Chatbot Endpoint
# -----------------------------
@app.post("/chat")
async def medical_chatbot(request: ChatRequest):
    zephyr_prompt = build_zephyr_prompt(request.prompt)

    try:
        response = client.text_generation(
            model=MODEL_NAME,
            prompt=zephyr_prompt,
            max_new_tokens=200,
            temperature=0.4,
            do_sample=True,
            return_full_text=False
        )

        clean_response = re.sub(
            r"<think>.*?</think>",
            "",
            response,
            flags=re.DOTALL
        ).strip()

        return {
            "reply": clean_response
        }

    except Exception as e:
        return {"error": str(e)}

# -----------------------------
# Health check
# -----------------------------
@app.get("/")
def root():
    return {"message": "Medical Chatbot API is running!"}

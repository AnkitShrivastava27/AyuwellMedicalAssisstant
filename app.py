import os
import google.generativeai as genai
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

# Configure Gemini
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise RuntimeError("GOOGLE_API_KEY not found.")

genai.configure(api_key=api_key)

# SYSTEM INSTRUCTION: Medical Assistant Role
# We include instructions for empathy, clarity, and safety.
SYSTEM_PROMPT = (
    "Role: You are a helpful, empathetic, and knowledgeable Medical Assistant. "
    "Task: Answer health-related queries, explain medical conditions in simple terms, "
    "and provide general wellness advice based on established medical guidelines. "
    "Guidelines: "
    "1. Always include a disclaimer that you are an AI, not a doctor. "
    "2. If symptoms sound life-threatening (e.g., chest pain, difficulty breathing), "
    "immediately advise the user to call emergency services. "
    "3. Use Markdown for clarity (bullet points for symptoms, bold text for key terms). "
    "4. Avoid overly dense jargon; explain things like a helpful peer. "
    "5. Be concise but thorough."
)

app = FastAPI(title="Gemini Medical Assistant API")

# Initializing Gemini 2.0 Flash (Stable 2026)
model = genai.GenerativeModel(
    
    model_name='gemini-2.5-flash-lite',
    system_instruction=SYSTEM_PROMPT
)

class MedicalQuery(BaseModel):
    prompt: str

@app.post("/generate")
async def medical_consult(request: MedicalQuery):
    if not request.prompt.strip():
        raise HTTPException(status_code=400, detail="Please enter your symptoms or question.")

    try:
        # Use a slightly higher temperature (0.7) for a more natural, empathetic tone
        response = model.generate_content(
            request.prompt,
            generation_config={"temperature": 0.7, "max_output_tokens": 2048}
        )

        if not response.text:
            return {"response": "I'm sorry, I cannot provide information on that specific topic due to safety guidelines."}

        return {"response": response.text.strip()}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Assistant Error: {str(e)}")
import os
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_community.chat_models import AzureChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
from langchain.memory import ConversationSummaryBufferMemory

# Load environment variables
load_dotenv()

# Initialize FastAPI
app = FastAPI(title="Medical Assistant API", version="1.0")

# CORS setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],       # You can replace with your frontend URL for security
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# LLM setup
llm_model = AzureChatOpenAI(
    azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_KEY"),
    api_version=os.getenv("AZURE_OPENAI_VERSION"),
    temperature=0.3,
    top_p=0.9,
    max_tokens=300,
)

# Prompt template
chat_prompt = ChatPromptTemplate.from_template("""
You are a professional, empathetic, and knowledgeable medical assistant.
Always provide medically accurate, clear, and concise information.
If the question is outside medical scope, politely decline.
If you don't know the answer, say "I don't know" instead of making up information.
If you answer the question, then at last add a new line:
In case of severe problem please consult a doctor.

{chat_history}
Patient's question: {question}
Answer:
""")

# Memory
memory = ConversationSummaryBufferMemory(
    llm=llm_model,
    max_token_limit=1000,
    return_messages=True,
    input_key="question",
    output_key="text",
    memory_key="chat_history",
)

# LLM chain
llm_chain = LLMChain(
    llm=llm_model,
    prompt=chat_prompt,
    memory=memory,
)

# Request model
class ChatRequest(BaseModel):
    question: str

# Accept both /chat and /chat/ to prevent 405 errors
@app.post("/chat")
@app.post("/chat/")
async def chat(chat_request: ChatRequest):
    question = chat_request.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    try:
        result = llm_chain.invoke({"question": question})
        return {
            "user": question,
            "assistant": result["text"],
            "memory_summary": memory.buffer
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM error: {str(e)}")

# Root health check
@app.get("/test")
async def root():
    return {"message": "Medical Assistant API is running."}

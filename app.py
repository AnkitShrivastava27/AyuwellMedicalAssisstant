import os
from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel
from langchain_community.chat_models import AzureChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
from langchain.memory import ConversationSummaryBufferMemory

# Load environment variables from .env
load_dotenv()

# Initialize FastAPI app
app = FastAPI(title="Medical Assistant API", version="1.0")

# Initialize LLM only once
llm_model = AzureChatOpenAI(
    azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_KEY"),
    api_version=os.getenv("AZURE_OPENAI_VERSION"),
    temperature=0.3,  # Lower for factual accuracy
    top_p=0.9,
    max_tokens=300,
)

# Prompt template
chat_prompt = ChatPromptTemplate.from_template("""
You are a professional, empathetic, and knowledgeable medical assistant.
Always provide medically accurate, clear, and concise information.
If the question is outside medical scope, politely decline.

{chat_history}
Patient's question: {question}
Answer:
""")

# Memory to store conversation
memory = ConversationSummaryBufferMemory(
    llm=llm_model,
    max_token_limit=1000,
    return_messages=True,
    input_key="question",
    output_key="text",
    memory_key="chat_history",
)

# Create LLM chain
llm_chain = LLMChain(
    llm=llm_model,
    prompt=chat_prompt,
    memory=memory,
)

# Request model
class ChatRequest(BaseModel):
    question: str

# POST endpoint for chat
@app.post("/chat")
def chat(request: ChatRequest):
    result = llm_chain.invoke({"question": request.question})
    return {
        "user": request.question,
        "assistant": result["text"],
        "memory_summary": memory.buffer
    }

# GET endpoint for health check
@app.get("/")
def root():
    return {"message": "Medical Assistant API is running."}

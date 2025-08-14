import os
import streamlit as st
from dotenv import load_dotenv
from langchain_community.chat_models import AzureChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
from langchain.memory import ConversationSummaryBufferMemory

# Load environment variables
load_dotenv()

# Initialize session state on first load
if "memory" not in st.session_state:
    # LLM setup
    llm_model = AzureChatOpenAI(
        azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_key=os.getenv("AZURE_OPENAI_KEY"),
        api_version=os.getenv("AZURE_OPENAI_VERSION"),
        temperature=0.7,
        top_p=0.9,
        max_tokens=100,
    )

    # Prompt
    chat_prompt = ChatPromptTemplate.from_template("""
You are a helpful assistant.
{chat_history}
User question: {question}
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

    # Chain
    llm_chain = LLMChain(
        llm=llm_model,
        prompt=chat_prompt,
        memory=memory,
    )

    st.session_state.llm_chain = llm_chain
    st.session_state.memory = memory
    st.session_state.chat_history = []

# Streamlit UI
st.title("💬 Chat with Azure GPT")

user_input = st.text_input("Ask something...", key="input")

if user_input:
    result = st.session_state.llm_chain.invoke({"question": user_input})
    st.session_state.chat_history.append(("You", user_input))
    st.session_state.chat_history.append(("Assistant", result["text"]))

# Display chat history
for speaker, msg in st.session_state.chat_history:
    st.markdown(f"**{speaker}:** {msg}")

# Optional: show summarized memory
with st.expander("🔍 View memory summary"):
    st.write(st.session_state.memory.buffer)
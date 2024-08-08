import os
import time
import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain_together import Together
from langchain.memory import ConversationBufferWindowMemory
from langchain.chains import ConversationalRetrievalChain

st.set_page_config(page_title="AILA")

# Initialize session state
if "show_chat" not in st.session_state:
    st.session_state.show_chat = False
if "messages" not in st.session_state:
    st.session_state.messages = []
if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferWindowMemory(k=2, memory_key="chat_history", return_messages=True)

# Function to initialize the chatbot components
def initialize_chatbot():
    embeddings = HuggingFaceEmbeddings(model_name="nomic-ai/nomic-embed-text-v1", model_kwargs={"trust_remote_code": True, "revision": "289f532e14dbbbd5a04753fa58739e9ba766f3c7"})
    db = FAISS.load_local("ipc_vector_db", embeddings, allow_dangerous_deserialization=True)
    db_retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 4})

    prompt_template = """<s>[INST]This is a chat template and As a legal chat bot specializing in Indian Penal Code queries, your primary objective is to provide accurate and concise information based on the user's questions. 
    Also mention the act and section numbers according to IPC  .Do not generate your own questions and answers. You will adhere strictly to the instructions provided, offering relevant context from the knowledge base while avoiding unnecessary details. Your responses will be brief, to the point, and in compliance with the established format. If a question falls outside the given context, you will refrain from utilizing the chat history and instead rely on your own knowledge base to generate an appropriate response. You will prioritize the user's query and refrain from posing additional questions. The aim is to deliver professional, precise, and contextually relevant information pertaining to the Indian Penal Code.
    CONTEXT: {context}
    CHAT HISTORY: {chat_history}
    QUESTION: {question}
    ANSWER:
    </s>[INST]
    """
    prompt = PromptTemplate(template=prompt_template, input_variables=['context', 'question', 'chat_history'])

    llm = Together(
        model="mistralai/Mistral-7B-Instruct-v0.2",
        temperature=0.5,
        max_tokens=1024,
        together_api_key="829b64678275cfdd5fbc043ea2fe54ff6404aa82381b3b11ef69617c91f132d5"
    )

    qa = ConversationalRetrievalChain.from_llm(
        llm=llm,
        memory=st.session_state.memory,
        retriever=db_retriever,
        combine_docs_chain_kwargs={'prompt': prompt}
    )

    return qa

# Function to reset the conversation
def reset_conversation():
    st.session_state.messages = []
    st.session_state.memory.clear()

# Function to show the initial description page
def show_description():
    st.title("AI Legal Advisor")
    st.image("AILA.jpg")
    st.markdown("""
        Welcome to AI-Legal-Advisor (AILA), an AI-powered legal assistant specializing in Indian Penal Code queries. 
        This tool provides accurate and concise information based on your questions, including act numbers according to IPC 
        and previous judgments on similar situations if available.
        Feel free to put your query and get a satisfactory result.
    """)
    st.markdown(""" """)
    st.button("Try Now", on_click=lambda: st.session_state.update({"show_chat": True}))

# Show the description or chatbot based on session state
if not st.session_state.show_chat:
    show_description()
else:
    qa = initialize_chatbot()
    st.image("AILA.jpg")
    for message in st.session_state.messages:
        with st.chat_message(message.get("role")):
            st.write(message.get("content"))

    input_prompt = st.chat_input("Enter your Query : ")

    if input_prompt:
        with st.chat_message("user"):
            st.write(input_prompt)

        st.session_state.messages.append({"role": "user", "content": input_prompt})

        with st.chat_message("assistant"):
            with st.status("Thinking 💡...", expanded=True):
                result = qa.invoke(input=input_prompt)
                message_placeholder = st.empty()
                full_response = "⚠️ **_Note: Information provided may be inaccurate._** \n\n\n"
                for chunk in result["answer"]:
                    full_response += chunk
                    time.sleep(0.02)
                    message_placeholder.markdown(full_response + " ▌")
                st.button('Reset All Chat 🗑️', on_click=reset_conversation)

        st.session_state.messages.append({"role": "assistant", "content": result["answer"]})

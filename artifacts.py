import streamlit as  st
import os
import google.generativeai as genai
from langchain.prompts import PromptTemplate
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq



def checkfile_exist(file):
    file_name = file.name
    file_path = os.path.join('./uploaded_data', file_name)
    do_embedding = None
    if os.path.exists(file_path):        
        exist_warning_placeholder = st.empty()
        exist_warning_placeholder.warning(f"File {file_name} already exists", icon="⚠️")

        radio_placeholder = st.empty()
        with radio_placeholder: 
            overwrite = st.radio("Do you want to overwrite the file?\n\nThis will cause re-embedding and will taketime.", ("Yes", "No"), index=None, key='radio')
            if overwrite == "Yes":
                with open(file_path, "wb") as f:
                    f.write(file.getbuffer())
                st.success(f"File {file.name} has been saved successfully!", icon="✅")
                do_embedding = "YES"

            elif overwrite == 'No':
                st.info("File upload was canceled.", icon="ℹ️")
                do_embedding = "NO"

            if overwrite:
                del st.session_state['radio']
                exist_warning_placeholder.empty()

    else:
        with open(file_path, "wb") as f:
            f.write(file.getbuffer())
            do_embedding = "YES"

    return file_path, do_embedding


def save_embedding(docs, file_name):
    embedding = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
    placeholder = st.empty()
    placeholder.image("./webartifacts/progress.gif", use_container_width=True)
    db = FAISS.from_documents(docs, embedding=embedding)
    embedding_folder = os.path.join("./","embedding", file_name)
    os.makedirs(embedding_folder, exist_ok=True)
    db.save_local(folder_path=embedding_folder)
    placeholder.empty()
    st.write("Embedding Saved")


def get_prompt_template():
    contextual_prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""
    You are an AI assistant. Answer the following question **only** using the provided context.
    
    Context:
    {context}
    
    Question: {question}
    
    If the answer is not found in the context, respond with: "Information not available in the provided context."
    """
    )

    
    return contextual_prompt


def get_retriever(file_name):
    embedding = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
    embedding_folder = os.path.join("./","embedding", file_name)
    db = FAISS.load_local(embedding_folder, embeddings=embedding, allow_dangerous_deserialization=True)
    retriever = db.as_retriever()
    return retriever


def get_available_embedding():
    lst = os.listdir("./embedding/")
    lst.remove('.gitkeep')
    print(lst)
    return lst



def print_history():
    user_style = """
        background-color: #DCF8C6;
        padding: 10px;
        color: #000;
        border-radius: 10px;
        text-align: right;
        margin: 5px 0;
    """

    assistant_style = """
        background-color: #F1F0F0;
        padding: 10px;
        color: #000;
        border-radius: 10px;
        text-align: left;
        margin: 5px 0;
    """


    for chat in st.session_state.chat_history:
        if chat["role"] == "user":
            st.markdown(
                f'<div style="{user_style}"><b>👤 You:</b> {chat["content"]}</div>',
                unsafe_allow_html=True
            )            
            
        else:  # For assistant messages
            st.markdown(
                f'<div style="{assistant_style}"><b>🤖 Gemini:</b> {chat["content"]}</div>',
                unsafe_allow_html=True
            )


    return user_style, assistant_style

def get_groq_session(api_key):
    chat = ChatGroq(
                model="llama-3.3-70b-versatile",
                temperature=0,
                max_tokens=None,
                timeout=None,
                max_retries=2,
                api_key=api_key
            )

    return chat
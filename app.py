import streamlit as st
import os
import torch
from PyPDF2 import PdfReader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_docling import DoclingLoader
from artifacts import checkfile_exist, save_embedding, get_prompt_template
from artifacts import get_retriever, get_available_embedding, print_history, get_groq_session
from dotenv import load_dotenv

load_dotenv()

GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
GROQ_API_KEY = os.getenv('GROQ_API_KEY')


torch.classes.__path__ = []
st.set_page_config(page_title='GROQ powered Doc QA')
st.header("GROQ powered Doc QA")

saved_file_path = None
do_emedding = None
file_name = None
retriever = None
selected_folder = None

with st.sidebar:
    if st.button("Clear Chat History/Session", icon='🗑️'):
        print("Inside clear chat history")
        del st.session_state['chat_history']
        del st.session_state['chat_session']

if 'file_name' not in st.session_state:
    st.session_state.file_name = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
    
if "chat_session" not in st.session_state:
    st.session_state.prompt_template = get_prompt_template()
    st.session_state.chat_session = get_groq_session(GROQ_API_KEY)
    st.session_state.chain = st.session_state.prompt_template | st.session_state.chat_session

if retriever not in st.session_state:
    st.session_state.retriever = None


with st.sidebar:
    uploaded_file = st.file_uploader("Upload your file", accept_multiple_files=False)
    selfile = st.empty()

    if not uploaded_file:
        selected_folder = selfile.selectbox('Select a file:', get_available_embedding(), index=None)

    if uploaded_file:
        selfile.empty()
        if (st.session_state.file_name != uploaded_file.name) or (not st.session_state.file_name):
            file_name = uploaded_file.name
            st.session_state.file_name = file_name
            saved_file_path, do_emedding = checkfile_exist(uploaded_file)
            print(saved_file_path, do_emedding)

    elif selected_folder:
        if st.session_state.file_name != selected_folder or (not st.session_state.file_name):
            st.session_state.file_name = selected_folder


if st.session_state.file_name and do_emedding=="YES":
    file_name = st.session_state.file_name
    docs = None
    with st.sidebar:
        st.write("Document Processing")
        placeholder = st.empty()
        placeholder.image("./webartifacts/progress.gif", use_container_width=True)
        loader = DoclingLoader(saved_file_path)
        docs = loader.load()
        if len(docs) == 0:
            st.write("Invalid Doc Please check")
        placeholder.empty()
        st.write("Document Processed. \nEmbedding Started...")
        save_embedding(docs=docs, file_name=file_name)
        st.progress(100)
    


print(st.session_state.file_name, selected_folder, uploaded_file)
if st.session_state.file_name and (selected_folder or uploaded_file):
    user_style, assistant_style = print_history()
    input = st.chat_input("Ask You query")
    if not st.session_state.retriever:
        st.session_state.retriever = get_retriever(st.session_state.file_name)

    if input:
        print(input)
        st.markdown(f'<div style="{user_style}"><b>👤 You:</b> {input}</div>', unsafe_allow_html=True)

        retrieved_docs = st.session_state.retriever.invoke(input=input)
        retrieved_text = "\n\n".join([doc.page_content for doc in retrieved_docs])
        formatted_prompt = st.session_state.prompt_template.format(context=retrieved_text, question=input)
        response = st.session_state.chat_session.invoke(formatted_prompt)
        response = response.content
        
        st.markdown(f'<div style="{assistant_style}"><b>🤖 Gemini:</b> {response}</div>', unsafe_allow_html=True)    
        
        st.session_state.chat_history.append({"role": "user", "content": input})
        st.session_state.chat_history.append({"role": "assistant", "content": response})
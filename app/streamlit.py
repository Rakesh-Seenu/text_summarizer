#!/usr/bin/env python3

'''
Talk2Biomodels: A Streamlit app for the Talk2Biomodels graph.
'''

import os
import sys
import random
import streamlit as st
from streamlit_feedback import streamlit_feedback
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.messages import ChatMessage
from langchain_openai import ChatOpenAI
from utils import streamlit_utils
from text_summarizer.tools.text_summazier import TextSummarizerTool  # Import the summarizer tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
import chardet  # Import chardet for detecting file encoding

st.set_page_config(page_title="Talk2Biomodels", page_icon="🤖", layout="wide")
# Set the logo

# Check if env variables OPENAI_API_KEY and/or
# NVIDIA_API_KEY exist
if "OPENAI_API_KEY" not in os.environ:
    st.error("Please set the OPENAI_API_KEY in the .env file.")
    st.stop()

# Import the agent
sys.path.append('./')
from text_summarizer.agents.t2b_agent import get_app

########################################################################################
# Streamlit app
########################################################################################
# Create a chat prompt template
prompt = ChatPromptTemplate.from_messages([
        ("system", "Welcome to Talk2Biomodels!"),
        MessagesPlaceholder(variable_name='chat_history', optional=True),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
])

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Initialize sbml_file_path
if "sbml_file_path" not in st.session_state:
    st.session_state.sbml_file_path = None

# Initialize project_name for Langsmith
if "project_name" not in st.session_state:
    # st.session_state.project_name = str(st.session_state.user_name) + '@' + str(uuid.uuid4())
    st.session_state.project_name = 'T2B-' + str(random.randint(1000, 9999))

# Initialize run_id for Langsmith
if "run_id" not in st.session_state:
    st.session_state.run_id = None

# Initialize graph
if "unique_id" not in st.session_state:
    st.session_state.unique_id = random.randint(1, 1000)
if "app" not in st.session_state:
    if "llm_model" not in st.session_state:
        st.session_state.app = get_app(st.session_state.unique_id,
                                llm_model=ChatOpenAI(model='gpt-4o-mini', temperature=0))
    else:
        print(st.session_state.llm_model)
        st.session_state.app = get_app(st.session_state.unique_id,
                                llm_model=streamlit_utils.get_base_chat_model(
                                st.session_state.llm_model))

if "app" not in st.session_state:
    st.session_state.app = get_app(st.session_state.unique_id,
                                   llm_model=ChatOpenAI(model='gpt-4o-mini', temperature=0))
# Get the app
app = st.session_state.app

prompt = ChatPromptTemplate.from_messages([
    ("system", "Welcome to Talk2Biomodels!"),
    MessagesPlaceholder(variable_name='chat_history', optional=True),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}"),
])


@st.fragment
def get_uploaded_files():
    """
    Upload files (PDF, DOCX, Text).
    """
    # Upload the article or text file
    article = st.file_uploader(
        "Upload an article (PDF, DOCX, or Text)",
        help="Upload a PDF, DOCX, or plain text file for summarization.",
        accept_multiple_files=False,
        type=["pdf", "docx", "txt"],
        key="article"
    )

    if article:
        import tempfile
        from PyPDF2 import PdfReader
        from docx import Document

        file_extension = article.name.split('.')[-1].lower()
        temp_file_path = tempfile.mktemp()

        with open(temp_file_path, 'wb') as temp_file:
            temp_file.write(article.getbuffer())

        st.session_state.sbml_file_path = temp_file_path  # Save the file path in session state

        # Process based on the file type
        try:
            if file_extension == "txt":
                # Try reading the file with multiple encodings
                try:
                    with open(temp_file_path, 'r', encoding='utf-8') as file:
                        file_content = file.read()
                except UnicodeDecodeError:
                    try:
                        with open(temp_file_path, 'r', encoding='ISO-8859-1') as file:
                            file_content = file.read()
                    except UnicodeDecodeError:
                        with open(temp_file_path, 'r', encoding='cp1252', errors='ignore') as file:
                            file_content = file.read()

                st.session_state.sbml_file_content = file_content

            elif file_extension == "pdf":
                # Read PDF file using PyPDF2
                reader = PdfReader(temp_file_path)
                pdf_content = ""
                for page in reader.pages:
                    pdf_content += page.extract_text() or ""
                st.session_state.sbml_file_content = pdf_content

            elif file_extension == "docx":
                # Read DOCX file using python-docx
                doc = Document(temp_file_path)
                doc_content = ""
                for para in doc.paragraphs:
                    doc_content += para.text + "\n"
                st.session_state.sbml_file_content = doc_content

            else:
                st.error("Unsupported file type!")
                return None

            # Display the content preview
            st.write(f"File uploaded successfully! Here's a preview of the content: {st.session_state.sbml_file_content[:500]}...")

        except UnicodeDecodeError as e:
            st.error(f"Error reading the file: {e}")
        except Exception as e:
            st.error(f"Unexpected error: {e}")
        
        return article

# Main layout of the app split into two columns
main_col1, main_col2 = st.columns([3, 7])
# First column
with main_col1:
    st.write("### 🤖 Talk2Biomodels")
    llms = ["OpenAI/gpt-4o-mini", "NVIDIA/llama-3.3-70b-instruct"]
    st.selectbox("Pick an LLM", llms, index=0, key="llm_model", help="Used for tool calling and generating responses.")
    
    # File upload
    uploaded_article = get_uploaded_files()

# Second column: chat history and user input
with main_col2:
    st.write("### 💬 Chat History")

    # Display history of messages
    for message in st.session_state.messages:
        if message["type"] == "message":
            # Check if 'message' content is a dictionary, use 'get()' to access keys
            content = message.get("content", {})
            role = content.get("role", "")
            text = content.get("content", "")
            
            with st.chat_message(role, avatar="🤖" if role != 'user' else "👩🏻‍💻"):
                st.markdown(text)
            st.empty()

    prompt_input = st.chat_input("Ask something ...", key="st_chat_input")

# Handling user input
if prompt_input:
    if uploaded_article:
        # Ensure the uploaded article is correctly processed
        with open(st.session_state.sbml_file_path, 'r', encoding='utf-8') as file:
            st.session_state.sbml_file_path = file.read()

        # Use the TextSummarizerTool to summarize the uploaded file content
        text_summarizer = TextSummarizerTool()
        summarized_content = text_summarizer.summarize(st.session_state.sbml_file_content)

        # Display the summarized content in the chat history
        st.write("### Summary of the Document")
        st.write(summarized_content)

        # Optionally, include the summarized content as input for further processing
        prompt_input = f"Here is the summary of the uploaded document:\n{summarized_content}\n{prompt_input}"

    prompt_msg = {"type": "message", "content": {"role": "user", "content": prompt_input}}
    st.session_state.messages.append(prompt_msg)

    with st.chat_message("user", avatar="👩🏻‍💻"):
        st.markdown(prompt_input)

    with st.chat_message("assistant", avatar="🤖"):
        with st.spinner("Fetching response ..."):
            # Get chat history
            history = [(m["content"].get("role", ""), m["content"].get("content", "")) for m in st.session_state.messages if m["type"] == "message"]

            # Send to the app and get response
            response = app.stream(
                {"messages": [HumanMessage(content=prompt_input)]},
                config={"configurable": {"thread_id": st.session_state.unique_id}},
                stream_mode="messages"
            )
            st.write_stream(streamlit_utils.stream_response(response))
            assistant_msg = {"role": "assistant", "content": response}
            st.session_state.messages.append({"type": "message", "content": assistant_msg})

import streamlit as st
from text_summarizer.document_processor import DocumentProcessor
from text_summarizer.chatbot import Chatbot
from text_summarizer.graph import ChatGraph

st.set_page_config(page_title="AI Chatbot with Documents", layout="wide")
st.title("📄 AI Chatbot with Document Uploads")

processor = DocumentProcessor()
chatbot = Chatbot()
chat_graph = ChatGraph().build_graph()

uploaded_files = st.file_uploader("Upload PDFs, DOCX, or CSV files", accept_multiple_files=True)

context = ""
if uploaded_files:
    for file in uploaded_files:
        context += processor.process_file(file) + "\n"

st.text_area("Extracted Document Content (Preview)", context[:1000])  # Show a preview

st.subheader("💬 Chat with the Bot")
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

user_input = st.text_input("Ask a question:")
if st.button("Send"):
    if user_input:
        output = chat_graph.invoke({"user_input": user_input, "context": context})
        response = output["response"]
        
        st.session_state.chat_history.append(("You", user_input))
        st.session_state.chat_history.append(("Bot", response))

for role, msg in st.session_state.chat_history:
    st.markdown(f"**{role}:** {msg}")

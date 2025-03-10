import os
from langchain.chat_models import ChatOpenAI
from langchain.schema import AIMessage, HumanMessage
from langchain.memory import ConversationBufferMemory

class Chatbot:
    def __init__(self, model_name="gpt-4"):
        """Initialize the chatbot with memory and an OpenAI LLM."""
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("Missing OpenAI API Key. Set OPENAI_API_KEY as an environment variable.")

        self.llm = ChatOpenAI(model_name=model_name, openai_api_key=api_key)
        self.memory = ConversationBufferMemory(return_messages=True)

    def chat(self, user_input, context=""):
        """Handles chat with memory & document context."""
        history = self.memory.load_memory_variables({})["history"]
        history.append(HumanMessage(content=f"Context: {context}\nUser: {user_input}"))

        response = self.llm(history)
        self.memory.save_context({"input": user_input}, {"output": response.content})

        return response.content

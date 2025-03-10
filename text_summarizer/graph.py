import os
from dotenv import load_dotenv 
from langgraph.graph import StateGraph, END
from langchain.schema import SystemMessage, HumanMessage, AIMessage
from langchain.chat_models import ChatOpenAI

# Define the schema as a list of tuples (the correct format)
state_schema = [
    ("user_input", str),
    ("context", str),
    ("response", str)
]

class ChatGraph:
    def __init__(self):
        # Fetch the OpenAI API key from environment variables
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if openai_api_key is None:
            raise ValueError("OpenAI API key not found. Please set the OPENAI_API_KEY in the .env file.")
        
        # Initialize ChatOpenAI with the API key
        self.llm = ChatOpenAI(model_name="gpt-4", openai_api_key=openai_api_key)

    def answer_question(self, state):
        """Handles user queries with context from documents and memory."""
        user_input = state["user_input"]
        context = state["context"]

        messages = [
            SystemMessage(content="You are a helpful AI assistant."),
            HumanMessage(content=f"Context: {context}\nUser: {user_input}")
        ]

        response = self.llm(messages)
        return {"response": response.content}

    def fallback(self, state):
        """Fallback method if initial answer is weak."""
        user_input = state["user_input"]
        messages = [HumanMessage(content=f"Can you clarify your question? You asked: {user_input}")]
        response = self.llm(messages)
        return {"response": response.content}

    def build_graph(self):
        """Creates a modular LangGraph conversation flow."""
        graph = StateGraph(state_schema=state_schema)

        graph.add_node("answer_question", self.answer_question, input=["user_input", "context"], output=["response"])
        graph.add_node("fallback", self.fallback, input=["user_input"], output=["response"])

        graph.add_edge("answer_question", END)
        graph.add_edge("fallback", END)

        graph.set_entry_point("answer_question")
        return graph.compile()

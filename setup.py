from setuptools import setup, find_packages

setup(
    name="text_summarizer_chatbot",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "streamlit",
        "langchain",
        "langgraph",
        "pypdf",
        "PyPDF2",
        "python-docx",
        "pandas",
        "openai"
    ],
    entry_points={
        "console_scripts": [
            "chatbot-app=text_summarizer_chatbot.app:main"
        ]
    },
)

#!/usr/bin/env python3

"""
Tool for summarizing text.
"""

import logging
from typing import Type, Annotated
from pydantic import BaseModel, Field
from langgraph.types import Command
from langgraph.prebuilt import InjectedState
from langchain_core.tools import BaseTool
from langchain_core.messages import ToolMessage
from langchain_core.tools.base import InjectedToolCallId
from transformers import pipeline  # Using Hugging Face's transformers for text summarization

# Initialize logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define the input schema for the text summarization tool
class TextSummarizationInput(BaseModel):
    """
    Input schema for the TextSummarization tool.
    """
    text_to_summarize: str = Field(..., description="The text that needs to be summarized")
    tool_call_id: Annotated[str, InjectedToolCallId]
    state: Annotated[dict, InjectedState]

# Note: It's important that every field has type hints. BaseTool is a
# Pydantic class and not having type hints can lead to unexpected behavior.
class TextSummarizerTool(BaseTool):
    """
    Tool for summarizing text.
    """
    name: str = "summarizer_tool"
    description: str = "A tool to summarize text"
    args_schema: Type[BaseModel] = TextSummarizationInput

    def _run(self,
        tool_call_id: Annotated[str, InjectedToolCallId],
        state: Annotated[dict, InjectedState],
        text_to_summarize: str = None
    ) -> Command:
        """
        Run the text summarization tool.

        Args:
            tool_call_id (str): The tool call ID. This is injected by the system.
            state (dict): The state of the tool.
            text_to_summarize (str): The text to be summarized.

        Returns:
            str: The summarized text.
        """
        logger.info("Running text summarization tool...")

        # Load Hugging Face summarization pipeline
        summarizer = pipeline("summarization")

        # Perform summarization
        summary = summarizer(text_to_summarize, max_length=200, min_length=50, do_sample=False)
        
        # Extract the summarized text from the result
        summarized_text = summary[0]['summary_text']
        
        logger.info("Text summarization complete.")
        
        # Prepare the result for returning as part of the tool's output
        dic_updated_state_for_model = {
            "summarized_text": summarized_text,
            "tool_call_id": tool_call_id
        }

        # Return the updated state of the tool, including the summary
        return Command(
            update=dic_updated_state_for_model | {
                # Update the message history with the summary
                "messages": [
                    ToolMessage(
                        content=f"Summarized text: {summarized_text}",
                        tool_call_id=tool_call_id
                    )
                ]
            }
        )

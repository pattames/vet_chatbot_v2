import os
from typing import List, Dict
from crewai import Agent, Task, Crew, Process
from langchain_groq import ChatGroq
import logging

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initiazile Groq LLM
llm = ChatGroq(
    model="groq/llama-3.3-70b-versatile",
    temperature=0.3,
    api_key=os.getenv("GROQ_API_KEY")
)

import os
from typing import List, Dict
from crewai import Agent, Task, Crew, Process
from langchain_groq import ChatGroq
import logging
from pydantic import SecretStr

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Groq LLM
api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    logger.error("GROQ_API_KEY environment variable is required")
    raise ValueError("GROQ_API_KEY environment variable is required")

llm = ChatGroq(
    model="groq/llama-3.3-70b-versatile",
    temperature=0.3,
    api_key=SecretStr(api_key)
)

# ==========================================
# AGENTS DEFINITION
# ==========================================
class VeterinaryAgents:
    """Define all agents for the veterinary chatbot system"""

    def classification_agent(self) -> Agent:
        """Agent that classifies queries"""
        return Agent(
            role="Médico veterinario especializado en identificar consultas de tipo veterinarias o no veterinarias",
            goal="Identificar si la consulta del usuario es de tipo veterinaria o no veterinaria",
            backstory="Eres un médico veterinario con 5 años de experiencia en la clasificación de casos. Tienes la habilidad de identificar de manera precisa si la consulta es de tipo veterinaria o no veterinaria.",
            llm=llm,
            verbose=True
        )

# =========================================
# TASKS DEFINITION
# =========================================
class VeterinaryTasks:
    """Define all tasks for the veterinary chatbot workflow"""

    def classification_task(self, agent: Agent, user_query:str) -> Task:
        """Classify query"""
        return Task(
            description=f"""Analiza esta consulta: '{user_query}' y determina si es de tipo A, B o C.
            - A: si la consulta se relaciona con medicina veterinaria
            - B: si la consulta busca un seguimiento a la conversación mantenida hasta este punto y la conversación se relaciona con medicina veterinaria (revisar en memoria)
            - C: si la consulta no se relaciona con medicina veterinaria""",
            expected_output="Solamente la letra A, B o C, dependiendo del tipo del resultado.",
            agent=agent,
        )

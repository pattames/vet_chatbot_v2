import os
from typing import List, Dict
from crewai import Agent, Task
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
    model="groq/llama-3.3-70b-versatile", temperature=0.3, api_key=SecretStr(api_key)
)


# ==========================================
# AGENTS DEFINITION
# ==========================================
class VeterinaryAgents:
    """Define all agents for the veterinary chatbot system"""

    def classification_agent(self) -> Agent:
        """Agent that classifies queries"""
        return Agent(
            role="Especialista en Triaje Veterinario",
            goal="Clasificar consultas de usuarios para determinar si requieren atención veterinaria, son seguimientos conversacionales, o están fuera del ámbito veterinario",
            backstory="""Eres un veterinario con experiencia en recepción de clínicas veterinarias.
Has procesado miles de consultas y desarrollaste intuición para identificar rápidamente qué tipo de atención necesita cada caso. Valoras la eficiencia y la precisión en el triaje inicial.""",
            llm=llm,
            verbose=True,
        )

    def veterinary_specialist_agent(self) -> Agent:
        """Agent that formulates responses"""
        return Agent(
            role="Veterinario Clínico",
            goal="Proporcionar respuestas veterinarias precisas y apropiadas",
            backstory="""Eres un veterinario clínico senior con más de 15 años de experiencia.
Eres excelente explicando conceptos complejos de manera clara y siempre priorizas tanto la seguridad del paciente como la precisión médica.""",
            llm=llm,
            verbose=True,
            allow_delegation=False
    )


# =========================================
# TASKS DEFINITION
# =========================================
class VeterinaryTasks:
    """Define all tasks for the veterinary chatbot workflow"""

    def classification_task(self, agent: Agent, user_query: str, conversation_history: List[Dict[str, str]]) -> Task:
        """Classify query"""

        # Recent conversation history
        recent_history = "Sin historial"
        if conversation_history:
            recent_history = "\n".join(
                [f"{msg['role']}: {msg['content']}" for msg in conversation_history[-6:]]
            )

        return Task(
            description=f"""Clasifica la siguiente consulta del usuario.

Consulta: '{user_query}'

Historial de conversación reciente:
{recent_history}

Tipos de clasificación:
- A: La consulta trata directamente sobre medicina veterinaria (síntomas, tratamientos, cuidados animales)
- B: La consulta es un seguimiento al historial de conversación (referencias a "eso", "lo anterior", preguntas de continuidad)
- C: La consulta no tiene relación con medicina veterinaria

Analiza la consulta y responde ÚNICAMENTE con la letra correspondiente.""",
            expected_output="Solamente la letra A, B o C, dependiendo del tipo del resultado.",
            agent=agent,
        )

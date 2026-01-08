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

# Different temperature LLMs to suit task
classification_llm = ChatGroq(model="groq/llama-3.3-70b-versatile", temperature=0.1, api_key=SecretStr(api_key))    # More deterministic
specialist_llm = ChatGroq(model="groq/llama-3.3-70b-versatile", temperature=0.4, api_key=SecretStr(api_key))    # More natural

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
            llm=classification_llm,
            verbose=True,
        )

    def specialist_agent(self) -> Agent:
        """Agent that formulates responses"""
        return Agent(
            role="Veterinario Clínico",
            goal="Proporcionar respuestas veterinarias precisas y apropiadas, o redirigir amablemente consultas fuera del ámbito veterinario",
            backstory="""Eres un veterinario clínico senior con más de 15 años de experiencia.
Eres excelente explicando conceptos complejos de manera clara y siempre priorizas tanto la seguridad del paciente como la precisión médica.""",
            llm=specialist_llm,
            verbose=True,
        )

# =========================================
# TASKS DEFINITION
# =========================================
class VeterinaryTasks:
    """Define all tasks for the veterinary chatbot workflow"""

    # Conversation history properly formatted (last three interactions as a single string)
    def _formatted_history(self, conversation_history: List[Dict[str, str]], limit: int = 6) -> str:
        if not conversation_history:
            return "Sin historial"
        return "\n".join(
            [f"{msg['role']}: {msg['content']}" for msg in conversation_history[-limit:]]
        )

    def classification_task(self, agent: Agent, user_query: str, conversation_history: List[Dict[str, str]]) -> Task:
        """Classify query"""

        formatted_history = self._formatted_history(conversation_history)
        return Task(
            description=f"""Clasifica la siguiente consulta del usuario.

Consulta: {user_query}

Historial de conversación reciente:
{formatted_history}

Tipos de clasificación:
- A: La consulta trata directamente sobre medicina veterinaria (síntomas, tratamientos, cuidados animales)
- B: La consulta es un seguimiento al historial de conversación (referencias a "eso", "lo anterior", preguntas de continuidad)
- C: La consulta no tiene relación con medicina veterinaria

Analiza la consulta y responde ÚNICAMENTE con la letra correspondiente.""",
            expected_output="Solamente la letra A, B o C, dependiendo del tipo del resultado.",
            agent=agent,
        )

    def response_task(self, agent: Agent, user_query: str, conversation_history: List[Dict[str, str]], context: List[Task]) -> Task:
        """Formulate appropriate response based on query type"""

        formatted_history = self._formatted_history(conversation_history)
        return Task(
            description=f"""Formula una respuesta apropiada basándote en el tipo de consulta identificado.

Consulta del usuario: {user_query}

Historial de conversación reciente:
{formatted_history}

Instrucciones según el tipo de clasificación:
- Tipo A (consulta veterinaria directa): Proporciona una respuesta médica precisa y útil.
- Tipo B (seguimiento): Considera el historial y continúa la conversación de manera coherente.
- Tipo C (no veterinaria): Indica amablemente que solo puedes responder consultas relacionadas con veterinaria.""",
            agent=agent,
            expected_output="Respuesta completa y apropiada para el tipo de consulta",
            context=context
        )

# ==============================
# CREW ORCHESTRATION
# ==============================
class VeterinaryCrue:
    """Orchestrate the multi-agent veterinary chatbot workflow"""

    def __init__(self):
        self.agent_manager = VeterinaryAgents()
        self.task_manager = VeterinaryTasks()

    def run(self, user_query: str, conversation_history: List[Dict[str, str]]):
        """Execute the multi-agent workflow for a user query"""

        logger.info(f"Processing query: {user_query}")

        # Initialize agents
        classification_agent = self.agent_manager.classification_agent()
        specialist_agent= self.agent_manager.specialist_agent()

        # Initialize tasks with dependencies
        classification_task = self.task_manager.classification_task(classification_agent, user_query, conversation_history)
        response_task = self.task_manager.response_task(specialist_agent, user_query, conversation_history, context=[classification_task])

        # Initialize crew, run it and return the result
        crew = Crew(
            agents=[classification_agent, specialist_agent],
            tasks=[classification_task, response_task],
            process=Process.sequential,
            stream=False,
            verbose=True
        )

        logger.info("Query processing completed")
        return crew.kickoff()

# =============================
# MAIN EXECUTION (for testing)
# =============================
if __name__ == "__main__":
    logger.error("Unable to run directly without Streamlit app running")
    logger.info("main.py won't work without Streamlit's conversation history")


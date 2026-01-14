# Veterinary AI Assistant V2

Multi-agent chatbot using CrewAI and Streamlit.

## Prerequisites

- Python 3.13+
- pip

## Setup

1. **Clone the repository:**

   ```bash
   git clone <repository-url>
   cd vet_chatbot_v2
   ```

2. **Create a virtual environment:**

   ```bash
   python3 -m venv .venv
   ```

3. **Activate the virtual environment:**

   - On macOS/Linux:
     ```bash
     source .venv/bin/activate
     ```
   - On Windows:
     ```bash
     .venv\Scripts\activate
     ```

4. **Install dependencies:**

   ```bash
   pip install -r requirements.lock -r requirements-dev.lock
   ```

5. **Create environment variables file:**

   ```bash
   # copy .env.example content and paste to .env while creating it
   cp .env.example .env
   ```

   Then edit `.env` and add your API keys:

   ```bash
   # Edit with your preferred editor
   vim .env  # or nvim, nano, code, etc.
   ```

6. **Start the app:**

   ```bash
   streamlit run app.py
   ```

   The UI will open in your browser at `http://localhost:8501`

## When Adding or Removing Dependencies

1. **Add or remove them from** `pyproject.toml`

   Then run:

   ```bash
   pip install -e .[dev]
   ```

2. **Regenerate lock files**

   ```bash
   pip-compile -o requirements.lock pyproject.toml
   pip-compile --extra dev -o requirements-dev.lock pyproject.toml
   ```

3. **Ensure your environment exactly matches both lock files**

   ```bash
   pip-sync requirements.lock requirements-dev.lock
   ```

## Deactivating Virtual Environment

When done working on the project:

```bash
deactivate
```

## Troubleshooting

- If you encounter module import errors, ensure the virtual environment is activated
- For "command not found" errors, verify all dependencies are installed
- Check that Python version is 3.13 or higher: `python --version`

"""Developer Support Agent - a Microsoft Foundry learning project.

Modules:
    config         Environment loading and validation.
    tools          Custom function tools and their JSON schemas.
    conversation   Shared conversation/reply data types.
    agent          Agent definition, local simulation backend and CLI.
    foundry_agent  Real Microsoft Foundry backend (requires Azure).
    ollama_agent   Local LLM backend using Ollama.
    api            REST API (FastAPI) over the same agent; run ``python -m app.api``.
"""

__version__ = "1.0.0"

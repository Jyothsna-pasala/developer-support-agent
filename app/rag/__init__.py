"""Document search (Retrieval-Augmented Generation) for the Developer Support Agent.

Uploaded PDF, DOCX and TXT files are split into chunks, embedded with an
Ollama embedding model and stored in PostgreSQL with pgvector, so the agent
can search them. Disabled unless ``RAG_ENABLED=true``.

Modules:
    settings  ``RAG_*`` environment settings and their validation.
"""

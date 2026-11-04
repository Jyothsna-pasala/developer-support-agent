"""REST API (FastAPI) for the Developer Support Agent.

Modules:
    main      FastAPI application factory and routes.
    sessions  Thread-safe in-memory sessions and turn concurrency limits.
    schemas   Request and response models.
    settings  ``API_*`` environment settings (host, port, CORS, limits).

Run with ``python -m app.api``. Local development only: no authentication.
"""

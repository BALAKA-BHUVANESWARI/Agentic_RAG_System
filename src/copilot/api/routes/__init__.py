"""API route modules."""
from copilot.api.routes import audit, chat, documents, evaluation, feedback, health

ALL_ROUTERS = [health.router, chat.router, documents.router, feedback.router, audit.router, evaluation.router]

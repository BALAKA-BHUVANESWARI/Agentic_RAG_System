# IT Support Copilot — Private-First Agentic RAG

Local demo project: answers employee IT questions from internal documents first, falls back to
public web search (Tavily) only when graded evidence is insufficient, cites every claim, abstains
when evidence is weak, and records the full decision path.

> All documents in `data/kb_documents/` are **synthetic demo documents, not real company policies**.
> Authorization is a **local role demo (employee/admin), not production identity management**.

**Status:** Phase 1 (foundation) complete. Full README (architecture, evaluation results, limitations) arrives in Phase 8.

## Setup (Windows, VS Code terminal)

```powershell
cd it-support-copilot
py -3.12 -m venv .venv          # any Python 3.11+ works; check installed versions with: py -0
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install -e .
copy .env.example .env           # then fill in GROQ / PINECONE / TAVILY keys
python scripts\check_env.py
python scripts\init_db.py
pytest tests\unit
```

## Layout
See `CLAUDE.md` for the directory layout, routing invariants and development rules.

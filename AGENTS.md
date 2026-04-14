# AGENTS.md

- Prefer minimal, working changes over large refactors.
- Reuse existing EDF and FastAPI code where possible.
- Keep model loading isolated in its own module.
- Do not invent checkpoints; only use real public checkpoints with a source URL.
- Update README with exact commands and limitations.
- Ensure the app can run locally with uvicorn.
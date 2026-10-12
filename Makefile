.PHONY: install test backend frontend lint format

install:
	uv sync --frozen
	cd frontend && npm ci

test:
	.venv/bin/python -m pytest -v

backend:
	.venv/bin/uvicorn backend.app.main:app --host 0.0.0.0 --port 8100 --reload

frontend:
	cd frontend && npm run dev

lint:
	cd frontend && npx tsc --noEmit

format:
	cd frontend && npx prettier --write app components

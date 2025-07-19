.PHONY: help install dev download run test clean docker-build docker-run docker-stop

# Default target
help:
	@echo "LogPrompt - Simple API for Transformer Models"
	@echo ""
	@echo "Available commands:"
	@echo "  install     - Install dependencies using pipenv"
	@echo "  dev         - Install development dependencies"
	@echo "  download    - Download all supported models"
	@echo "  run         - Run the FastAPI application"
	@echo "  test        - Run tests"
	@echo "  clean       - Clean up cache and temporary files"
	@echo "  docker-build - Build Docker image"
	@echo "  docker-run  - Run with Docker Compose"
	@echo "  docker-stop - Stop Docker containers"
	@echo "  docker-logs - View Docker logs"
	@echo "  format      - Format code with black"
	@echo "  autopep8    - Format code with autopep8"
	@echo "  lint        - Lint code with flake8"
	@echo "  typecheck   - Type check with mypy"
	@echo "  quality     - Run all quality checks (format, lint, typecheck)"
	@echo "  pre-commit  - Install pre-commit hooks"
	@echo "  check       - Run quality checks and tests"

# Python/Pipenv commands
install:
	pipenv install

dev:
	pipenv install --dev

download:
	pipenv run python download_models.py

run:
	pipenv run uvicorn main:app --host 0.0.0.0 --port 8000 --reload

test:
	pipenv run pytest

format:
	pipenv run black .

autopep8:
	pipenv run autopep8 --in-place --recursive .

lint:
	pipenv run flake8 .

typecheck:
	pipenv run mypy .

quality: format lint typecheck
	@echo "All quality checks completed!"

pre-commit:
	pipenv run pre-commit install
	@echo "Pre-commit hooks installed!"

pre-commit-run:
	pipenv run pre-commit run --all-files

clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf .pytest_cache
	rm -rf build/
	rm -rf dist/

# Docker commands
docker-build:
	docker-compose build

docker-run:
	docker-compose up -d

docker-stop:
	docker-compose down

docker-logs:
	docker-compose logs -f

docker-restart:
	docker-compose restart

# Combined commands
setup: install
	@echo "Creating models directory..."
	mkdir -p models
	mkdir -p logs
	@echo "Setup complete!"

dev-setup: dev setup
	@echo "Development environment ready!"

# Code quality
check: quality test
	@echo "All checks passed!"

# Production deployment
deploy: docker-build docker-run
	@echo "Application deployed successfully!"
	@echo "API available at: http://localhost:8000"
	@echo "API docs available at: http://localhost:8000/docs"

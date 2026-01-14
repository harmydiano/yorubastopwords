.PHONY: help install run test clean docker-build docker-run lint format

help:
	@echo "Yoruba Stopwords Generator - Available Commands:"
	@echo "  make install       - Install dependencies"
	@echo "  make run          - Run the application locally"
	@echo "  make test         - Run unit tests"
	@echo "  make lint         - Run code linters"
	@echo "  make format       - Format code with black"
	@echo "  make clean        - Clean temporary files"
	@echo "  make docker-build - Build Docker image"
	@echo "  make docker-run   - Run Docker container"
	@echo "  make docker-stop  - Stop Docker container"

install:
	pip install --upgrade pip
	pip install -r requirements.txt

run:
	python app.py

test:
	pytest tests/ -v --cov=.

lint:
	flake8 *.py --max-line-length=120 --exclude=venv,env
	mypy *.py --ignore-missing-imports

format:
	black *.py --line-length=120

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache .coverage htmlcov
	rm -f output*.txt

docker-build:
	docker build -t yoruba-stopwords .

docker-run:
	docker-compose up -d

docker-stop:
	docker-compose down

docker-logs:
	docker-compose logs -f

setup-dev:
	cp .env.example .env
	mkdir -p uploads logs
	pip install -r requirements.txt
	@echo "Development environment setup complete!"
	@echo "Edit .env file with your configuration"

# LogPrompt - Simple API for Transformer Models

A simple web service for running BERT and RoBERTa transformer models with automatic downloading and local caching.

## Supported Models

-   BERT-base-uncased
-   BERT-large-uncased
-   RoBERTa-base
-   RoBERTa-large
-   ALBERT-base-v1
-   ALBERT-base-v2

## Features

-   🚀 FastAPI-based web service
-   🤖 Automatic model downloading and caching
-   💾 Local model storage
-   🐳 Docker and Docker Compose support
-   📦 Pipenv for dependency management
-   🧪 Unit tests included
-   📊 Built-in health checks

## Quick Start

### Pre-download Models (Recommended)

For better performance, download all models before starting the API:

```bash
# Install dependencies
make install

# Download all supported models (this may take some time)
make download

# Run the application
make run
```

### Using Make (Recommended)

```bash
# Install dependencies
make install

# Run the application (models will be downloaded on first use)
make run
```

### Using Pipenv Directly

```bash
# Install dependencies
pipenv install

# Run the application
pipenv run uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

### Using Docker

```bash
# Build and run with Docker Compose
make docker-run

# Or manually
docker-compose up -d
```

## API Usage

### Start the Server

The API will be available at `http://localhost:8000`

### API Endpoints

#### 1. Get API Information

```bash
curl http://localhost:8000/
```

#### 2. List Models

```bash
curl http://localhost:8000/models
```

#### 3. Health Check

```bash
curl http://localhost:8000/health
```

#### 4. Generate Embeddings

```bash
curl -X POST "http://localhost:8000/predict"
     -H "Content-Type: application/json"
     -d '{
       "text": "Hello, this is a test sentence.",
       "model_name": "bert-base-uncased",
       "task": "feature-extraction"
     }'
```

#### 5. Preload a Model

```bash
curl -X POST "http://localhost:8000/load-model/roberta-base"
```

### Example Python Client

```python
import requests

# API endpoint
url = "http://localhost:8000/predict"

# Request data
data = {
    "text": "This is an example sentence for processing.",
    "model_name": "bert-base-uncased",
    "task": "feature-extraction"
}

# Send request
response = requests.post(url, json=data)
result = response.json()

print(f"Model: {result['model_name']}")
print(f"Embeddings shape: {len(result['embeddings'])} x {len(result['embeddings'][0])}")
```

## Model Management

### Download All Models

The `download_models.py` script allows you to pre-download all supported models with progress tracking:

```bash
# Using make
make download

# Or directly
pipenv run python download_models.py
```

Features of the download script:

-   📊 **Progress bars** for each model and overall progress
-   🔄 **Resume support** - skips already downloaded models
-   📁 **Organized storage** - models stored in `./models/` directory
-   ✅ **Error handling** - continues if individual models fail
-   🖥️ **CUDA detection** - automatically detects GPU availability

### Model Storage

Models are stored locally in the `./models/` directory structure:

```
models/
├── bert-base-uncased/
├── bert-large-uncased/
├── roberta-base/
├── roberta-large/
├── albert-base-v1/
└── albert-base-v2/
```

### Runtime Model Loading

If models are not pre-downloaded, they will be downloaded automatically on first use with progress indicators.

## Development

### Setup Development Environment

```bash
make dev-setup
```

### Run Tests

```bash
make test
```

### Code Formatting

```bash
# Format with Black
make format

# Format with autopep8
make autopep8
```

### Linting

```bash
make lint
```

### Type Checking

```bash
make typecheck
```

### All Quality Checks

```bash
make quality
```

### Pre-commit Hooks

```bash
# Install pre-commit hooks
make pre-commit

# Run pre-commit on all files
make pre-commit-run
```

## Project Structure

```
├── main.py              # FastAPI application
├── test_main.py         # Unit tests
├── test_api.py          # API integration tests
├── Pipfile              # Python dependencies
├── Dockerfile           # Docker configuration
├── docker-compose.yml   # Docker Compose setup
├── Makefile            # Build and run commands
├── models/             # Local model cache (auto-created)
└── logs/               # Application logs (auto-created)
```

## Configuration

### Environment Variables

-   `TRANSFORMERS_CACHE`: Cache directory for models (default: `./models`)
-   `PYTHONPATH`: Python path (default: `/app` in Docker)

### Model Storage

Models are automatically downloaded and cached in the `./models` directory. The first request for each model will take longer due to downloading.

## Docker Usage

### Build and Run

```bash
# Build the image
make docker-build

# Run with Docker Compose
make docker-run

# View logs
make docker-logs

# Stop containers
make docker-stop
```

### Manual Docker Commands

```bash
# Build
docker build -t logprompt-api .

# Run
docker run -p 8000:8000 -v $(pwd)/models:/app/models logprompt-api
```

## API Documentation

Once the server is running, visit:

-   Swagger UI: `http://localhost:8000/docs`
-   ReDoc: `http://localhost:8000/redoc`

## Requirements

-   Python 3.12
-   4GB+ RAM (recommended for larger models)
-   Internet connection (for initial model downloads)

## Performance Notes

-   First request per model will be slower due to downloading
-   Subsequent requests use cached models for faster response
-   GPU support available if CUDA is detected
-   Models are loaded lazily (on first request)

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests: `make test`
5. Format code: `make format`
6. Submit a pull request

## License

See LICENSE file for details.

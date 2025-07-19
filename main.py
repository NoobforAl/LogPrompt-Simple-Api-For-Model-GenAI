#!/usr/bin/env python3
"""
LogPrompt - Simple API for Transformer Models
A FastAPI web service for running BERT and RoBERTa models
"""

import os
import logging
from typing import Dict, List, Optional, AsyncGenerator, Any
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModel, pipeline
import torch

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Model configuration
SUPPORTED_MODELS = {
    "bert-base-uncased": "bert-base-uncased",
    "bert-large-uncased": "bert-large-uncased",
    "roberta-base": "roberta-base",
    "roberta-large": "roberta-large",
    "albert-base-v1": "albert-base-v1",
    "albert-base-v2": "albert-base-v2"
}

# Local model storage directory
MODELS_DIR = "./models"

# Global model cache
model_cache: Dict[str, Dict[str, Any]] = {}


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan manager"""
    # Startup
    ensure_models_directory()

    # Check for pre-downloaded models
    available_models = []
    for model_name in SUPPORTED_MODELS.keys():
        model_path = os.path.join(MODELS_DIR, model_name)
        if os.path.exists(model_path) and os.listdir(model_path):
            available_models.append(model_name)

    logger.info("🚀 LogPrompt API starting...")
    logger.info(f"📁 Models directory: {os.path.abspath(MODELS_DIR)}")
    logger.info(f"🖥️  CUDA available: {torch.cuda.is_available()}")
    logger.info(
        f"📦 Pre-downloaded models: {len(available_models)}/{len(SUPPORTED_MODELS)}")

    if available_models:
        logger.info(f"✅ Available models: {', '.join(available_models)}")
    else:
        logger.warning("⚠️  No pre-downloaded models found!")
        logger.info("💡 Run 'python download_models.py' to download all models first")

    logger.info("✅ LogPrompt API started successfully!")
    yield
    # Shutdown
    logger.info("👋 LogPrompt API shutting down...")


# Create FastAPI app with lifespan
app = FastAPI(
    title="LogPrompt - Transformer Models API",
    description="Simple API for running BERT and RoBERTa models",
    version="1.0.0",
    lifespan=lifespan
)


class ModelRequest(BaseModel):
    text: str
    model_name: str
    task: str = "feature-extraction"  # Default task


class ModelResponse(BaseModel):
    model_name: str
    text: str
    task: str
    embeddings: Optional[List[List[float]]] = None
    features: Optional[Dict] = None
    error: Optional[str] = None


def ensure_models_directory() -> None:
    """Create models directory if it doesn't exist"""
    if not os.path.exists(MODELS_DIR):
        os.makedirs(MODELS_DIR)
        logger.info(f"Created models directory: {MODELS_DIR}")


def download_and_cache_model(model_name: str) -> Dict[str, Any]:
    """Download and cache a model if not already cached"""
    if model_name in model_cache:
        return model_cache[model_name]

    if model_name not in SUPPORTED_MODELS:
        raise ValueError(f"Model {model_name} not supported")

    model_id = SUPPORTED_MODELS[model_name]
    cache_dir = os.path.join(MODELS_DIR, model_name)

    try:
        logger.info(f"📥 Loading model: {model_name}")
        from tqdm import tqdm

        # Check if model exists locally first
        try:
            # Try to load from local cache first
            with tqdm(total=3, desc=f"🔧 Loading {model_name}", unit="step", ncols=80) as pbar:
                pbar.set_postfix_str("Loading tokenizer...")
                tokenizer = AutoTokenizer.from_pretrained(
                    cache_dir,
                    local_files_only=True
                )
                pbar.update(1)

                pbar.set_postfix_str("Loading model...")
                model = AutoModel.from_pretrained(
                    cache_dir,
                    local_files_only=True
                )
                pbar.update(1)

                pbar.set_postfix_str("Creating pipeline...")
                pipe = pipeline(
                    "feature-extraction",
                    model=model,
                    tokenizer=tokenizer,
                    device=0 if torch.cuda.is_available() else -1
                )
                pbar.update(1)
                pbar.set_postfix_str("✅ Loaded from cache")
            logger.info(f"✅ Loaded {model_name} from local cache")

        except Exception:
            # If local loading fails, download from hub
            logger.info(
                f"📥 Local cache not found, downloading {model_name} from Hugging Face Hub...")
            with tqdm(total=3, desc=f"⬇️  Downloading {model_name}", unit="step", ncols=80) as pbar:
                pbar.set_postfix_str("Downloading tokenizer...")
                tokenizer = AutoTokenizer.from_pretrained(
                    model_id,
                    cache_dir=cache_dir,
                    local_files_only=False
                )
                pbar.update(1)

                pbar.set_postfix_str("Downloading model...")
                model = AutoModel.from_pretrained(
                    model_id,
                    cache_dir=cache_dir,
                    local_files_only=False
                )
                pbar.update(1)

                pbar.set_postfix_str("Creating pipeline...")
                pipe = pipeline(
                    "feature-extraction",
                    model=model,
                    tokenizer=tokenizer,
                    device=0 if torch.cuda.is_available() else -1
                )
                pbar.update(1)
                pbar.set_postfix_str("✅ Downloaded")
            logger.info(f"✅ Downloaded and cached {model_name}")

        model_cache[model_name] = {
            "tokenizer": tokenizer,
            "model": model,
            "pipeline": pipe
        }
        return model_cache[model_name]

    except Exception as e:
        logger.error(f"❌ Error loading model {model_name}: {str(e)}")
        raise


@app.get("/")
async def root() -> Dict[str, Any]:
    """Root endpoint"""
    return {
        "message": "LogPrompt - Transformer Models API",
        "version": "1.0.0",
        "supported_models": list(SUPPORTED_MODELS.keys())
    }


@app.get("/models")
async def list_models() -> Dict[str, Any]:
    """List all supported models"""
    return {
        "supported_models": list(SUPPORTED_MODELS.keys()),
        "loaded_models": list(model_cache.keys())
    }


@app.post("/predict", response_model=ModelResponse)
async def predict(request: ModelRequest) -> ModelResponse:
    """Generate predictions using specified model"""
    try:
        # Validate model name
        if request.model_name not in SUPPORTED_MODELS:
            raise HTTPException(
                status_code=400, detail=f"Model {
                    request.model_name} not supported. Available models: {
                    list(
                        SUPPORTED_MODELS.keys())}")

        # Load model if not cached
        model_data = download_and_cache_model(request.model_name)

        # Generate embeddings/features
        pipeline_obj = model_data["pipeline"]

        if request.task == "feature-extraction":
            # Get embeddings
            embeddings = pipeline_obj(request.text)
            return ModelResponse(
                model_name=request.model_name,
                text=request.text,
                task=request.task,
                embeddings=embeddings
            )
        else:
            # For other tasks, return basic features
            tokenizer = model_data["tokenizer"]
            tokens = tokenizer.tokenize(request.text)

            return ModelResponse(
                model_name=request.model_name,
                text=request.text,
                task=request.task,
                features={
                    "num_tokens": len(tokens),
                    "tokens": tokens[:50],  # Limit tokens for response size
                    "text_length": len(request.text)
                }
            )

    except HTTPException:
        # Re-raise HTTP exceptions to maintain proper status codes
        raise
    except Exception as e:
        logger.error(f"❌ Error processing request: {str(e)}")
        # For non-HTTP exceptions, return error in response body with 200 status
        return ModelResponse(
            model_name=request.model_name,
            text=request.text,
            task=request.task,
            error=str(e)
        )


@app.post("/load-model/{model_name}")
async def load_model(model_name: str) -> Dict[str, str]:
    """Preload a specific model"""
    try:
        if model_name not in SUPPORTED_MODELS:
            raise HTTPException(
                status_code=400,
                detail=f"Model {model_name} not supported"
            )

        download_and_cache_model(model_name)
        return {"message": f"Model {model_name} loaded successfully"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check() -> Dict[str, Any]:
    """Health check endpoint"""
    return {
        "status": "healthy",
        "models_loaded": len(model_cache),
        "cuda_available": torch.cuda.is_available()
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

#!/usr/bin/env python3
"""
Model Download Script for LogPrompt API
Downloads all supported models before starting the API server
"""

import os
import sys
import logging
from typing import Dict, Any
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel, pipeline
import torch

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
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


def ensure_models_directory() -> None:
    """Create models directory if it doesn't exist"""
    if not os.path.exists(MODELS_DIR):
        os.makedirs(MODELS_DIR)
        logger.info(f"Created models directory: {MODELS_DIR}")


def download_model(model_name: str, model_id: str) -> Dict[str, Any]:
    """Download and cache a single model"""
    cache_dir = os.path.join(MODELS_DIR, model_name)

    try:
        logger.info(f"📥 Starting download for: {model_name}")

        # Create a progress bar for this model
        with tqdm(total=3, desc=f"🔧 {model_name}", unit="step", ncols=80) as pbar:
            # Download tokenizer
            pbar.set_postfix_str("Downloading tokenizer...")
            tokenizer = AutoTokenizer.from_pretrained(
                model_id,
                cache_dir=cache_dir,
                local_files_only=False
            )
            pbar.update(1)

            # Download model
            pbar.set_postfix_str("Downloading model...")
            model = AutoModel.from_pretrained(
                model_id,
                cache_dir=cache_dir,
                local_files_only=False
            )
            pbar.update(1)

            # Create pipeline
            pbar.set_postfix_str("Creating pipeline...")
            pipe = pipeline(
                "feature-extraction",
                model=model,
                tokenizer=tokenizer,
                device=0 if torch.cuda.is_available() else -1
            )
            pbar.update(1)
            pbar.set_postfix_str("✅ Complete")

        logger.info(f"✅ Successfully downloaded: {model_name}")
        return {
            "tokenizer": tokenizer,
            "model": model,
            "pipeline": pipe
        }

    except Exception as e:
        logger.error(f"❌ Error downloading {model_name}: {str(e)}")
        raise


def download_all_models() -> None:
    """Download all supported models"""
    ensure_models_directory()

    print("🚀 LogPrompt Model Downloader")
    print("=" * 50)
    print(f"📦 Downloading {len(SUPPORTED_MODELS)} models...")
    print(f"📁 Storage directory: {os.path.abspath(MODELS_DIR)}")
    print(f"🖥️  CUDA available: {torch.cuda.is_available()}")
    print()

    total_models = len(SUPPORTED_MODELS)
    failed_models = []

    # Overall progress bar
    with tqdm(total=total_models, desc="🌟 Overall Progress", unit="model", ncols=80) as overall_pbar:
        for i, (model_name, model_id) in enumerate(SUPPORTED_MODELS.items(), 1):
            try:
                print(f"\n📥 [{i}/{total_models}] Processing: {model_name}")
                download_model(model_name, model_id)
                overall_pbar.update(1)
                overall_pbar.set_postfix_str(f"✅ {model_name}")

            except Exception as e:
                failed_models.append((model_name, str(e)))
                overall_pbar.update(1)
                overall_pbar.set_postfix_str(f"❌ {model_name}")
                continue

    print("\n" + "=" * 50)
    if failed_models:
        print(f"❌ {len(failed_models)} models failed to download:")
        for model_name, error in failed_models:
            print(f"   • {model_name}: {error}")
        print(f"\n✅ {total_models - len(failed_models)} models downloaded successfully")
    else:
        print("🎉 All models downloaded successfully!")

    print(f"📁 Models stored in: {os.path.abspath(MODELS_DIR)}")
    print("🚀 You can now start the API server with: python main.py")


def main() -> None:
    """Main function"""
    try:
        download_all_models()
    except KeyboardInterrupt:
        print("\n⚠️  Download interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"❌ Download failed: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()

"""
hub_utils.py
-------------
A utility to map custom 'hf://' protocol URIs to local paths by downloading or
utilizing HuggingFace's internal caching system.
"""

import os
from pathlib import Path
import logging
from huggingface_hub import hf_hub_download

logger = logging.getLogger("layerlab.hub_utils")

def resolve_path(uri: str) -> Path:
    """
    Given a URI, returns a valid local pathlib.Path pointing to the file.
    If the URI starts with 'hf://', it downloads/caches the file from HuggingFace
    and returns its cached absolute path.
    Otherwise, assumes it is a standard physical path.
    """
    if uri.startswith("hf://"):
        # Format: hf://repo_id/filename
        # Example: hf://sshleifer/tiny-gpt2/model.safetensors
        # Remove prefix
        core = uri[5:]
        parts = core.split("/")
        if len(parts) < 2:
            raise ValueError(f"Invalid hf:// protocol signature: '{uri}'. Expected format: hf://repo_id/filename")
        
        filename = parts[-1]
        repo_id = "/".join(parts[:-1])
        
        logger.info(f"Intercepted hf:// scheme. Fetching '{filename}' from repo '{repo_id}'...")
        
        try:
            # Download or pull from local cache
            # This is synchronous but blocks harmlessly in FastAPI threadpools
            cached_path = hf_hub_download(repo_id=repo_id, filename=filename)
            logger.info(f"Successfully cached Model to {cached_path}")
            return Path(cached_path)
        except Exception as e:
            logger.error(f"HuggingFace Hub Retrieval Failed: {e}")
            raise RuntimeError(f"HuggingFace Hub Retrieval Failed: {e}")
            
    # Default fallback to standard local path evaluation
    return Path(uri)

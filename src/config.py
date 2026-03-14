"""Configuration loading and validation."""

from pathlib import Path
import yaml
from typing import Dict, Any


def load_config(config_path: str) -> Dict[str, Any]:
    """Load experiment configuration from YAML file.
    
    Args:
        config_path: Path to YAML configuration file
        
    Returns:
        Dictionary containing configuration
    """
    with open(Path(config_path), "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    
    # Validate required fields
    required_fields = ["experiment_name", "dataset", "task", "models", "output"]
    for field in required_fields:
        if field not in config:
            raise ValueError(f"Missing required field in config: {field}")
    
    # Pipeline is required only for embedding approach
    approach = config["task"].get("approach", "embedding")
    if approach == "embedding" and "pipeline" not in config:
        raise ValueError("Missing required field in config: pipeline (required for embedding approach)")
    
    return config


def get_model_config(config: Dict[str, Any], model_type: str) -> Dict[str, Any]:
    """Extract model configuration.
    
    Args:
        config: Full configuration dictionary
        model_type: Type of model ('biencoder' or 'reranker')
        
    Returns:
        Model configuration dictionary
    """
    return config["models"].get(model_type, {})
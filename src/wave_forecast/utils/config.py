import os
from omegaconf import OmegaConf

def load_config(config_path="config/production.yaml"):
    """
    Loads configuration from a YAML file.
    
    Args:
        config_path (str): Path to the configuration file.
    
    Returns:
        OmegaConf: A dictionary-like configuration object.
    """
    # Create a schema for validation if needed in the future
    # For now, just load the file
    cfg = OmegaConf.load(config_path)
    return cfg
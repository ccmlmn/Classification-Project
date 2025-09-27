import yaml
import os

def load_config():
    # Always resolve relative to project root (parent of src/)
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    config_path = os.path.join(base_dir, "config.yml")

    with open(config_path, "r") as file:
        return yaml.safe_load(file)

config_data = load_config()

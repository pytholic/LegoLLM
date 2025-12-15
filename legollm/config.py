from pathlib import Path

# Project root directory assuming config.py is in legollm
PROJECT_ROOT = Path(__file__).parents[1]
SOURCE_ROOT = PROJECT_ROOT / "legollm"

# Common paths
DATA_DIR = PROJECT_ROOT / "data"
TOKENIZERS_DIR = DATA_DIR / "tokenizers"
MODELS_DIR = SOURCE_ROOT / "models"

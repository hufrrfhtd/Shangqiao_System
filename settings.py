from pathlib import Path

from dotenv import load_dotenv

PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data"
RUNTIME_DIR = PROJECT_DIR / ".runtime"

load_dotenv(PROJECT_DIR / ".env")


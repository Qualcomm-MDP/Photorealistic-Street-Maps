import os
from pathlib import Path

from dotenv import load_dotenv


def app_env_file() -> str:
    """The env file for APP_ENV, as docker-compose.yml picks it:
    APP_ENV=development -> .env.development, unset -> .env."""
    env = os.environ.get("APP_ENV", "").strip()
    return f".env.{env}" if env else ".env"


def app_env_path() -> Path:
    """Where that env file lives: PSM_ENV_DIR (the project folder, when the
    web app runs in Docker), else the working directory."""
    return Path(os.environ.get("PSM_ENV_DIR") or ".") / app_env_file()


def load_app_env() -> None:
    """Load .env, or .env.<APP_ENV> (e.g. .env.development) when APP_ENV is set."""
    load_dotenv(app_env_path())

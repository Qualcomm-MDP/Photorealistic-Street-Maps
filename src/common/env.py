import os

from dotenv import load_dotenv


def load_app_env() -> None:
    """Load .env, or .env.development / .env.production when APP_ENV is set."""
    env = os.environ.get("APP_ENV", "").lower()
    env_file = {
        "development": ".env.development",
        "production": ".env.production",
    }.get(env, ".env")
    load_dotenv(env_file)

from dotenv import load_dotenv
import os

load_dotenv(dotenv_path="./.env", override=True)

OPENAI_TOKEN = os.getenv("OPENAI_API_KEY")
print(f"OPENAI_TOKEN: {OPENAI_TOKEN}")

PROJECT = os.getenv("PROJECT_NAME")
print(f"PROJECT NAME: {PROJECT}")
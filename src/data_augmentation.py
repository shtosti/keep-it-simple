from dotenv import load_dotenv
import os
from openai import OpenAI


load_dotenv(dotenv_path="./../.env", override=True)

OPENAI_TOKEN = os.getenv("OPENAI_API_KEY")
print(f"OPENAI_TOKEN: {OPENAI_TOKEN}")
PROJECT = os.getenv("PROJECT_NAME")
print(f"PROJECT NAME: {PROJECT}")


client = OpenAI(api_key=OPENAI_TOKEN)
completion = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {
            "role": "user",
            "content": "Simplify the first paragraph of the Bible for a 5-year-old."
        }
    ]
)

print(completion.choices[0].message)

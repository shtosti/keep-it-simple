import os
import json
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv(dotenv_path="./../.env", override=True)


OPENAI_TOKEN = os.getenv("OPENAI_API_KEY")
PROJECT = os.getenv("PROJECT_NAME")
MODEL = "gpt-4o-mini"
client = OpenAI(api_key=OPENAI_TOKEN)


def simplify_text(input_text, prompt, model=MODEL):
    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": f"Simplify this text: {input_text}"}
    ]
    
    # Generate completion
    response = client.chat.completions.create(
        model=model,
        messages=messages
    )
    return response.choices[0].message.content


def main():

    with open("./data/prompts/prompts.json", "r") as file:
        prompts = json.load(file)

    # TODO modify as needed
    prompt_category = "fkgl"
    prompt_id = "11"
    prompt = prompts[prompt_category][prompt_id]
    print(prompt)

    # TODO setup iteration over our datasets
    input_text = "In mammals, environmental sounds stimulate the auditory receptor, the cochlea, via vibrations of the stapes, the innermost of the middle ear ossicles."

    simplified_text = simplify_text(input_text, prompt)
    output_data = {
        "original_text": input_text,
        "simplified_text": simplified_text
    }
    print(output_data)

    # Save the result to a JSON file
    output_file = f"./output/{prompt_category}_{prompt_id}.json"
    with open(output_file, "w") as file:
        json.dump(output_data, file, indent=4)

    print(f"Simplified output saved to {output_file}")


if __name__ == "__main__":
    main()

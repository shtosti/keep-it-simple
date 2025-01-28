import os
import json
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv(dotenv_path="./../.env", override=True)


OPENAI_TOKEN = os.getenv("OPENAI_API_KEY")
PROJECT = os.getenv("PROJECT_NAME")
MODEL = "gpt-4o-mini"
client = OpenAI(api_key=OPENAI_TOKEN)


def generate_response(messages, model=MODEL):
    response = client.chat.completions.create(
        model=model,
        messages=messages
    )
    return response.choices[0].message.content

def simplify_text(input_text, prompt):
    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": f"Simplify this text: {input_text}"}
    ]
    response = generate_response(messages)

    return response

def simplify_text_with_reasoning(input_text, prompt):
    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": f"Briefly describe the needs of the target group, if such is specified. Using the instructions you were given, briefly explain how you would simplify the following source text and why: {input_text}. Think step by step, and very explain the changes you would make bake to satisfy the instructions, including any calculations, if required. Do not yet generate the simplification! Be very brief."}
    ]
    reasoning = generate_response(messages)

    messages.append({"role": "assistant", "content": reasoning})
    messages.append({"role": "user", "content": f"Now that you have developed a simplification strategy, generate the simplification. Only write the simplification. No other comments are allowed."})
    
    simplification = generate_response(messages)

    return reasoning, simplification

def simplify_text_consequtive_simplifications(input_text, prompt, model=MODEL):
    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": f"You will perform a step-by-step simplification of the following text: {input_text}."},
        {"role": "user", "content": f"Start by performing a syntactic simplification. Reduce sentences to minimal clauses. You can split sentences into several, if needed."}
    ]
    syntactic_symplification = generate_response(messages)

    messages.append({"role": "assistant", "content": syntactic_symplification})
    messages.append({"role": "user", "content": f"Proceed with a lexical simplification. Substitute domain-specific and difficult words with simple words, if possible."})

    lexical_simplification = generate_response(messages)

    messages.append({"role": "assistant", "content": lexical_simplification})
    messages.append({"role": "user", "content": f"Now, feel free to use paraphrase or explanation for terms and concepts you think require it."})

    paraphrase_or_explanation = generate_response(messages)

    messages.append({"role": "assistant", "content": paraphrase_or_explanation})
    messages.append({"role": "user", "content": f"Now, generate the final simplification based on your previous thoughts. Output only the simplification. No notes or comments are allowed."})

    simplification = generate_response(messages)

    return syntactic_symplification, lexical_simplification, paraphrase_or_explanation, simplification

def main():

    experiment_name = "basic" # with_reasoning, consequtive_simplifications, basic
    dataset_name = "sample_sentence_neuroscience"

    # create output dir
    OUTPUT_DIR = f"./output/prompting/{experiment_name}/{MODEL}/{dataset_name}"
    os.makedirs(os.path.dirname(OUTPUT_DIR), exist_ok=True)

    with open("./data/prompts/prompts.json", "r") as file:
        prompts = json.load(file)
        with open("./data/prompts/justifications.json", "r") as file:
            justifications = json.load(file)

    # TODO setup iteration over our datasets
    INPUT_TEXT = "Mice have long been a central part of neuroscience research, providing a flexible model that scientists can control and study to learn more about the intricate inner workings of the brain. Historically, researchers have favored male mice over female mice in experiments, in part due to concern that the hormone cycle in females causes behavioral variation that could throw off results."

    # Iterate through prompt categories and variations
    for category, variations in prompts.items():
        for prompt_id, prompt in variations.items():

            if experiment_name == "with_reasoning":
                reasoning, simplified_text = simplify_text_with_reasoning(INPUT_TEXT, prompt)
                output_data = {
                    "category": category,
                    "prompt_id": prompt_id,
                    "reasoning": reasoning,
                    "prompt_used": prompt,
                    "original_text": INPUT_TEXT,
                    "simplified_text": simplified_text,
                }
            elif experiment_name == "consequtive_simplifications":
                syntactic, lexical, paraphrase, simplified_text = simplify_text_consequtive_simplifications(INPUT_TEXT, prompt)
                output_data = {
                    "category": category,
                    "prompt_id": prompt_id,
                    "prompt_used": prompt,
                    "syntactic_simplification": syntactic,
                    "lexical_simplification": lexical,
                    "paraphrase_or_explanation": paraphrase,
                    "original_text": INPUT_TEXT,
                    "simplified_text": simplified_text,
                }
            else:
                simplified_text = simplify_text(INPUT_TEXT, prompt)
                output_data = {
                    "category": category,
                    "prompt_id": prompt_id,
                    "prompt_used": prompt,
                    "original_text": INPUT_TEXT,
                    "simplified_text": simplified_text
                }

            # Output file name
            output_file = f"{OUTPUT_DIR}/{category}_{prompt_id}.json"
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            with open(output_file, "w") as file:
                json.dump(output_data, file, indent=4)

            print(f"Output saved to {output_file}")


if __name__ == "__main__":
    main()
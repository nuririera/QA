import datetime
import time
from dataset_division import test_data
import requests
from collections import Counter
import re
import json

# Fecha para el nombre de archivo
date = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M")
global_start = time.time()

API_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "llama_ft"
N_RUNS = 5
MAX_RETRIES = 5
TIMEOUT = 30
NUM_PREDICT = 100
arguments = [entry["text"] for entry in test_data]

# Prompt idéntico al usado en el fine-tuning
prompt_intro = """
###ROLE### You are an Argument Annotator AI.

###OBJECTIVE### Your task is to asses the quality of an argument across four dimensions: cogency, effectiveness, reasonableness and overall. For each dimension, provide a binary score:
- "Bad"
- "Good"

###INSTRUCTIONS### You must evaluate each dimension independently, based strictly on the provided definitions.

Be particularly strict and conservative when evaluating. Do not hesitate to assign "Bad" if an argument does not clearly meet the criteria for that dimension.

The overall quality should reflect a synthesis of the other three dimensions but should also consider any other relevant factors.

Do not assume that most arguments are "Good". Your priority is to identify weaknesses and be sensitive to any lack of quality.

Return your response only as a JSON object using that values (Bad, Good). Do not use other labels.

You MUST ONLY return a single JSON object with exactly these four fields: cogency, effectiveness, reasonableness, overall. Values MUST ONLY be "Good" or "Bad", wrapped in double quotes.

DO NOT explain. DO NOT comment. DO NOT include any text before or after. ANY output not matching JSON format will be considered INVALID.

###DIMENSIONS###

1. **Cogency (Justification Quality)**
Evaluate only the justifications used to support the claim. Ask yourself:
- Are the justifications believable and relevant to the author's point?
- Do they provide enough support for the conclusion?

2. **Effectiveness (Persuasiveness and Presentation)**
Assess how persuasive the presentation is. Ask yourself:
- Is the author persuasive or credible?
- Does the argument evoke emotions appropriately?
- Is the language clear, appropriate and grammatically correct?
- Is the argument logically ordered and easy to follow?

3. **Reasonableness (Contribution to Issue Resolution)**
Consider the argument’s contribution to resolving the issue. Ask:
- Would the target audience accept it?
- Does it contribute meaningfully to the discussion?
- Does it provide helpful information for reaching a conclusion?
- Does it address counterarguments?

4. **Overall Quality**
Reflect on the three dimensions above. Consider any other relevant factors for the general quality of the argument.

###ARGUMENT###
"""

def build_prompt(argument):
    return f"{prompt_intro}\n{argument}\n###OUTPUT###"

def query_model(prompt):
    try:
        res = requests.post(
            API_URL,
            json={
                "model": MODEL_NAME,
                "prompt": prompt,
                "num_predict": NUM_PREDICT,
                "stream": False
            },
            timeout=TIMEOUT
        )

        if res.status_code != 200:
            print("Error from API:", res.text)
            return None

        return res.json().get("response", "{}")

    except requests.exceptions.Timeout:
        print("Request timed out after", TIMEOUT, "seconds.")
        return None

    except requests.exceptions.RequestException as e:
        print("Request failed:", e)
        return None


def extract_labels(text):
    try:
        match = re.search(r'(\{{1,2})(.*?)(\}{1,2})', text, re.DOTALL)
        if not match:
            print("No JSON-like object found in response.")
            return None
        
        json_text = match.group(0)

        if json_text.startswith('{{') and json_text.endswith('}}'):
            json_text = json_text[1:-1].strip()

        parsed = json.loads(json_text)
        expected_dims = ["cogency", "effectiveness", "reasonableness", "overall"]

        if all(dim in parsed for dim in expected_dims):
            return {dim: parsed[dim] for dim in expected_dims}
        else:
            return None

    except Exception as e:
        print("Error parsing response:", text)
        return None

error_counter = Counter()
all_runs = []

for run_ind in range(N_RUNS):
    run_start = time.time()
    print(f"\n--- RUN {run_ind + 1} ---")
    run = []
    local_errors = 0

    for i, arg in enumerate(arguments):
        retries = 0
        success = False

        while retries < MAX_RETRIES and not success:
            arg_start = time.time()
            prompt = build_prompt(arg)
            response = query_model(prompt)
            labels = extract_labels(response)

            if labels and all(dim in labels for dim in ["cogency", "effectiveness", "reasonableness", "overall"]):
                run.append(labels)
                success = True
            else:
                retries += 1
                local_errors += 1
                error_counter[f"arg_{i+1}_retry_{retries}"] += 1
                print(f"Retry {retries} for argument {i+1} due to invalid response.")
                time.sleep(1)

        if not success:
            print(f"Failed to process argument {i+1} after {MAX_RETRIES} retries. Skipping.")
            run.append(None)

        print(f"Argument {i + 1}:\n{arg}\nResponse: {run[-1]}\n")
        arg_time = time.time() - arg_start
        print(f"Time for argument {i + 1}: {arg_time:.2f} seconds")
        time.sleep(0.5)

    all_runs.append(run)

output_filename = f"model_responses_{date}.json"
with open(output_filename, "w") as f:
    json.dump(all_runs, f, indent=2)

print(f"\n--- RESPUESTAS GUARDADAS EN: {output_filename} ---")

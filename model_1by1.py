import datetime
import time
from dataset_division import test_data
import requests
from collections import Counter
import re
import json


#date in YYYY-MM-DD-HH-MM format
date = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M")
global_start = time.time() #total time

API_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "qwen3:8b"
N_RUNS = 3

arguments = [entry["text"] for entry in test_data]


dimensions_prompts = {
    "cogency": """
You are an Argument Quality Annotator.
Your task is to evaluate the **Cogency** of the followig argument.

Definition of Cogency (justification quality):
Evaluate only the justifications used to support the claim. Ask yourself:
- Are the justifications believable and relevant to the author's point?
- Do they provide enough support for the conclusion?

Return only one of these values in JSON, the values MUST be Good or Bad:

{
  "cogency": "Good" | "Bad"
}
""",
    "effectiveness": """
You are an Argument Quality Annotator.
Your task is to evaluate the **Effectiveness** of the followig argument.

Definition of Effectiveness (persuasiveness and presentation):
Assess how persuasive the presentation is. Ask yourself:
- Is the author persuasiive or credible?
- Does the argument evoke emotions appropriately?
- Is the language clear, appropiate and grammatically correct?
- Is the argument logically ordered and easy to follow?

Be strict and conservative. Return only one of these values in JSON, the values MUST be Good or Bad:

{
  "effectiveness": "Good" | "Bad"
}
""",
    "reasonableness": """
You are an Argument Quality Annotator.
Your task is to evaluate the **Reasonableness** of the followig argument.

Definition of Reasonableness (contribution to issue resolution):
Consider the argument’s contribution to resolving the issue. Ask:
- Would the target audience accept it?
- Does it contribute meaningfully to the discussion?
- Does it provide helpful information for reaching a conclusion?
- Does it address counterarguments?

Be strict and conservative. Return only one of these values in JSON, the values MUST be Good or Bad:

{
  "reasonableness": "Good" | "Bad"
}
""",
    "overall": """
You are an Argument Quality Annotator.
Your task is to evaluate the **Overall Quality** of the followig argument.

Return only one of these values in JSON, the values MUST be Good or Bad:

{
  "overall": "Good" | "Bad"
}
"""
}
# --- Prompt Builder según versión ---
def build_prompt_by_dimension(argument, dimension):
    return f"{dimensions_prompts[dimension]}\n\n###argument###\n{argument}###YOUR RESPONSE###"


# This function sends the prompt to the API and returns the response it also measures the response time
def query_model(prompt):
    try:
        res = requests.post(API_URL, json={
            "model": MODEL_NAME,
            "prompt": prompt,
            "stream": False
        })

        if res.status_code != 200:
            print("Error from API:", res.text)
            return None

        return res.json().get("response", "{}")
    
    except requests.exceptions.RequestException as e:
        print("Request failed:", e)
        return None
    
def extract_labels(text):
    try:
        match = re.search(r'\{.*?\}', text, re.DOTALL)
        if not match:
            print("No JSON found in response:", text)
            return None
        
        parsed = json.loads(match.group())
        expected_dims = ["cogency", "effectiveness", "reasonableness", "overall"]
        keys_present = [k for k in expected_dims if k in parsed]

        if len(keys_present) >= 1:
            return parsed
        else:
            return None

    except Exception as e:
        print("Error parsing response:", text)
        return None

    
MAX_RETRIES = 5
error_counter = Counter()
all_runs = []
for run_ind in range(N_RUNS):
    run_start = time.time()
    print(f"\n--- RUN {run_ind + 1} ---")
    run = []
    local_errors = 0

    for i, arg in enumerate(arguments):
        arg_start = time.time()
        labels = {}

        for dimension in ["cogency", "effectiveness", "reasonableness", "overall"]:
            dim_success = False
            retries = 0

            while retries < MAX_RETRIES and not dim_success:
                prompt = build_prompt_by_dimension(arg, dimension)
                response = query_model(prompt)
                dim_labels = extract_labels(response)

                if dim_labels and dimension in dim_labels:
                    labels[dimension] = dim_labels[dimension]
                    dim_success = True
                else:
                    retries += 1
                    local_errors += 1
                    error_counter[f"arg_{i+1}_{dimension}_retry_{retries}"] += 1
                    print(f"Retry {retries} for argument {i+1}, dimension {dimension} due to invalid response.")
                    time.sleep(1)

            if not dim_success:
                print(f"Failed to process argument {i+1}, dimension {dimension} after {MAX_RETRIES} retries. Skipping.")
                labels[dimension] = None  # marcador tipo 'None'

        run.append(labels)

        print(f"\nArgument {i + 1}:\n{arg}\nResponse: {labels}")
        arg_time = time.time() - arg_start
        print(f"Time for argument {i+1}: {arg_time:.3f} seconds")
        time.sleep(0.5)

    all_runs.append(run)
    print(f"\n--- Run {run_ind + 1} completed in {time.time() - run_start:.2f} seconds ---")

output_filename = f"model_responses_{date}.json"
with open(output_filename, "w") as f:
    json.dump(all_runs, f, indent=2)

print(f"\n--- SAVED RESPONSES IN: {output_filename} ---")
print(f"Total time: {time.time() - global_start:.2f} seconds")
print(f"Total local errors: {local_errors}")
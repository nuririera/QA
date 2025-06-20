import datetime
import sys
from Logger import Logger
import time
from dataset_division import test_data
import requests
from collections import Counter
import re
import json


#date in YYYY-MM-DD-HH-MM format
date = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M")

#Name of the file
log_filename = f"analysis_log_{date}.txt"

sys.stdout = Logger(log_filename)
global_start = time.time() #total time

API_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "llama3.1"
N_RUNS = 2

arguments = [entry["text"] for entry in test_data]

common_intro = """
####ROLE###
You are an Argument Annotator AI.

###OBJECTIVE###
Your task is to evaluate the quality of an argument in four dimensions: cogency, effectiveness, reasonableness, and overall.
You must score each of the four traits on a scale from 1 to 5:
- 1: Very Bad
- 2: Bad
- 3: Medium
- 4: Good
- 5: Very Good
Then, assign an overall quality score based on the other three.

Return your response only as a JSON object using numeric values (1 to 5). Do not use other labels.
"""

dimensions = """
#### DIMENSIONS & QUESTIONS ####

1. **Cogency**  
Evaluate only the justifications used to support the claim. Ask yourself:
- Are the justifications believable?
- Are they relevant to the author's point?
- Do they provide enough support for the conclusion?

2. **Effectiveness**  
Assess how persuasive the presentation is. Ask yourself:
- Is the author qualified or credible?
- Does the argument evoke emotions appropriately?
- Is the language clear and grammatically correct?
- Is the delivery appropriate for an online forum?
- Is the argument logically ordered and easy to follow?

3. **Reasonableness**  
Consider the argument’s contribution to resolving the issue. Ask:
- Would the target audience accept it?
- Does it contribute meaningfully to the discussion?
- Does it provide helpful information for reaching a conclusion?
- Does it address counterarguments?

4. **Overall Quality**  
Reflect on your previous scores. Consider any other relevant factors too.

"""

example = """
###EXPECTED OUTPUT###
Respond ONLY with a JSON object. The values MUST 1 or 2 or 3 or 4 or 5:
{{
  "cogency": 1 | 2 | 3 | 4 | 5,
  "effectiveness": 1 | 2 | 3 | 4 | 5,
  "reasonableness": 1 | 2 | 3 | 4 | 5,
  "overall": 1 | 2 | 3 | 4 | 5
}}


###EXAMPLE###
EXAMPLE argument:
Through cooperation, children can learn about interpersonal skills which are significant in the future life of all students.
What we acquired from team work is not only how to achieve the same goal with others but more importantly, how to get along with others.
During the process of cooperation, children can learn about how to listen to opinions of others, how to communicate with others, how to think comprehensively, and even how to compromise with other team members when conflicts occurred.
All of these skills help them to get on well with other people and will benefit them for the whole life.

EXAMPLE OUTPUT:
{{
    "cogency": 4,
    "effectiveness": 2,
    "reasonableness": 4,
    "overall": 4
}}
"""

# This function builds the prompt
def build_prompt(argument):
    return f"{common_intro}\n{dimensions}\n{example}\n\n###argument###\n{argument}###YOUR RESPONSE### (Only respond with the JSON object)"


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
    
def extract_labels(text, return_numeric=False):
    try:
        match = re.search(r'\{.*?\}', text, re.DOTALL)
        parsed = json.loads(match.group())

        expected_dims = ["cogency", "effectiveness", "reasonableness", "overall"]

        if all(dim in parsed for dim in expected_dims):
            return {dim: int(parsed[dim]) for dim in expected_dims}
        else:
            return None  # Missing keys

    except Exception as e:
        print("Error parsing response:", text)
        return None  # Invalid format
    
MAX_RETRIES = 5
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
            run.append(None)  # o gunmen marcador tipo 'None'

        print(f"Argument {i + 1}:\n{arg}\nResponse: {run[-1]}\n")
        arg_time = time.time() - arg_start
        print(f"Time for argument {i + 1}: {arg_time:.2f} seconds")
        time.sleep(0.5)  # optional cooldown

    all_runs.append(run)

output_filename = f"model_responses_{date}.json"
print("Contenido de all runs:")
print(json.dumps(all_runs, indent=2))
with open(output_filename, "w") as f:
    json.dump(all_runs, f, indent=2)

print(f"\n--- RESPUESTAS GUARDADAS EN: {output_filename} ---")
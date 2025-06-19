import requests
import re
import json
import time
from collections import Counter
from analyze_results import evaluate_single_run, analyze_variability_across_runs, evaluate_multiple_runs
from dataset_division import test_data
import sys
import time
from Logger import Logger

sys.stdout = Logger("analysis_log3.txt")
global_start = time.time() #total time

API_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "llama3.1"
N_RUNS = 10

arguments = [entry["text"] for entry in test_data]

common_intro = """
####ROLE###
You are an Argument Annotator AI.

###OBJECTIVE###
Given an argument, assess the quality.

###QUALITY RATINGS###
Assign a quality rating of either "Good" or "Bad" in each of the four categories. Internally, assign a value between 1-5 to each quality aspect. If the value is between 1-2.5, the quality is "Bad"; if it is between 2.5-5, the quality is "Good".
"""

dimensions = """
###DIMENSIONS OF ARGUMENT QUALITY###

The argument should be evaluated holistically according to four dimensions. First, consider logical cogency: whether the component presents ideas that are credible, relevant to the claim or conclusion, and sufficient to justify it. Second, assess rhetorical effectiveness by looking at how clearly the idea is expressed, whether the tone fits the topic and audience, how well it is structured, whether it adds to the author’s credibility, and whether it uses emotional appeal appropriately. Third, examine dialectical reasonableness, or the extent to which the argument is acceptable to the audience, contributes to resolving the issue, and addresses possible counterarguments. Finally, make an overall assessment: if the component performs well across most of these areas, label it <Good>; if not, label it <Bad>.
"""

example = """
###EXPECTED OUTPUT###
Resond in the following JSON format:
{{
"cogency": "Good" | "Bad",
"effectiveness": "Good" | "Bad",
"reasonableness": "Good" | "Bad",
"overall": "Good" | "Bad"
}}


###EXAMPLE###
EXAMPLE argument:
Through cooperation, children can learn about interpersonal skills which are significant in the future life of all students.
What we acquired from team work is not only how to achieve the same goal with others but more importantly, how to get along with others.
During the process of cooperation, children can learn about how to listen to opinions of others, how to communicate with others, how to think comprehensively, and even how to compromise with other team members when conflicts occurred.
All of these skills help them to get on well with other people and will benefit them for the whole life.

EXAMPLE OUTPUT:
{{
    "cogency": "Good",
    "effectiveness": "Bad",
    "reasonableness": "Good",
    "overall": "Good"
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

# Clean and parse the response
def extract_labels(text):
    try:
        match = re.search(r'\{.*?\}', text, re.DOTALL)
        parsed = json.loads(match.group())

        if all(dim in parsed for dim in ["cogency", "effectiveness", "reasonableness", "overall"]):
            return parsed
        else:
            return None  # Missing keys
    except Exception as e:
        print("Error parsing response:", text)
        return None  # Invalid format
    
# Multiple runs of the model
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
            run.append(None)  # o algún marcador tipo 'None'


        print(f"Argument {i + 1}:\n{arg}\nResponse: {run[-1]}\n")
        arg_time = time.time() - arg_start
        print(f"Time for argument {i + 1}: {arg_time:.2f} seconds")
        time.sleep(0.5)  # optional cooldown

    run_time = time.time() - run_start
    all_runs.append(run)
    print(f"Run {run_ind + 1} completed in {run_time:.2f} seconds.")
    print(f"Total errors in RUN {run_ind + 1}: {local_errors}")

ground_truth = [entry["labels"] for entry in test_data]

# Evaluate each run against the ground truth
for i, run in enumerate(all_runs):
    print(f"\n--- EVALUATION OF RUN {i + 1} ---")
    evaluate_single_run(run, ground_truth)

# Analyze variability across arguments
print("\n--- ANALYSIS OF VARIABILITY ACROSS ARGUMENTS ---")
evaluate_multiple_runs(all_runs, ground_truth)

# Analyze variability across runs
print("\n--- ANALYSIS OF VARIABILITY ACROSS RUNS ---")
analyze_variability_across_runs(all_runs)

# Print error statistics
print("\n--- ERROR STATISTICS ---")
print(f"Total errors across all runs: {sum(error_counter.values())}")
print("Error details:", dict(error_counter))

# Print total execution time
total_duration = time.time() - global_start
print(f"\n--- TOTAL EXECUTION TIME ---")
print(f"Total time: {total_duration:.2f} seconds")

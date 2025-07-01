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
MODEL_NAME = "llama3.1"
N_RUNS = 10
VERSION = 4 # chose between 4 versions

arguments = [entry["text"] for entry in test_data]

common_intro1 = """
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

common_intro2 = """
####ROLE###
You are an Argument Annotator AI.

###OBJECTIVE###
Your task is to evaluate the quality of an argument in four dimensions: cogency, effectiveness, reasonableness, and overall.
You must score each of the four traits on a scale:

- Bad
- Medium
- Good

Then, assign an overall quality score based on the other three.

Return your response only as a JSON object using that values (Bad, Medium, Good). Do not use other labels.
"""
common_intro3 = """
####ROLE###
You are an Argument Annotator AI.

###OBJECTIVE###
Your task is to asses the quality of an argument across four dimensions: cogency, effectiveness, reasonableness, and overall.
For each dimension, provide a binary score:
- "Bad"
- "Good"

You must evaluate each dimension independently, based strictly on the provided definitions.

Be particularly strict and conservative when evaluating. Do not hesitate to assign "Bad" if an argument does not clearly meet the criteria for that dimension.

The overall quality should reflect a synthesis of the other three dimensions but should also consider any other relevant factors.

Do not assume that most arguments are "Good". Your priority is to identify weaknesses and be sensitive to any lack of quality.

Return your response only as a JSON object using that values (Bad, Good). Do not use other labels.
"""
common_intro4 = """
####ROLE###
You are an Argument Annotator AI.

###OBJECTIVE###
Your task is to evaluate the quality of an argument in four dimensions: cogency, effectiveness, reasonableness, and overall.
You must score each of the four traits binary:
- Ineffective
- Effective
Then, assign an overall quality score based on the other three.

Return your response only as a JSON object using that values (Ineffective, Effective). Do not use other labels.
"""


dimensions = """
#### DIMENSIONS & QUESTIONS ####

1. **Cogency (Justification Quality)**  
Evaluate only the justifications used to support the claim. Ask yourself:
- Are the justifications believable and relevant to the author's point?
- Do they provide enough support for the conclusion?

2. **Effectiveness (Persuasiveness and Presentation)**  
Assess how persuasive the presentation is. Ask yourself:
- Is the author persuasiive or credible?
- Does the argument evoke emotions appropriately?
- Is the language clear, appropiate and grammatically correct?
- Is the argument logically ordered and easy to follow?

3. **Reasonableness (Contribution to Issue Resolution)**  
Consider the argument’s contribution to resolving the issue. Ask:
- Would the target audience accept it?
- Does it contribute meaningfully to the discussion?
- Does it provide helpful information for reaching a conclusion?
- Does it address counterarguments?

4. **Overall Quality**  
- Reflect on the three dimensions above.  
- Consider any other relevant factors for the general quality of the argument.

"""

example1 = """
###EXPECTED OUTPUT###
Respond ONLY with a JSON object. The values MUST be 1 or 2 or 3 or 4 or 5:
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
example2 = """
###EXPECTED OUTPUT###
Respond ONLY with a JSON object. The values MUST be "Good" or "Medium" or "Bad":
{{
  "cogency": "Good" | "Medium" | "Bad",
  "effectiveness": "Good" | "Medium" | "Bad",
  "reasonableness": "Good" | "Medium" | "Bad",
  "overall": "Good" | "Medium" | "Bad"
}}

Always wrap all values inside double quotes, so the output is always valid JSON.

###EXAMPLE###
EXAMPLE argument:
Through cooperation, children can learn about interpersonal skills which are significant in the future life of all students.
What we acquired from team work is not only how to achieve the same goal with others but more importantly, how to get along with others.
During the process of cooperation, children can learn about how to listen to opinions of others, how to communicate with others, how to think comprehensively, and even how to compromise with other team members when conflicts occurred.
All of these skills help them to get on well with other people and will benefit them for the whole life.

EXAMPLE OUTPUT:
{{
    "cogency": "Good",
    "effectiveness": "Medium",
    "reasonableness": "Bad",
    "overall": "Medium"
}}
"""

example3 = """
###EXPECTED OUTPUT###
Respond ONLY with a JSON object. The values MUST be Good or Bad:
{{
  "cogency": "Good" | "Bad",
  "effectiveness": "Good" | "Bad",
  "reasonableness": "Good" | "Bad",
  "overall": "Good" | "Bad"
}}

Always wrap all values inside double quotes, so the output is always valid JSON.



###EXAMPLE###
EXAMPLE argument:
Saying you belong to  a political ideology makes you dogmatic. Society is dynamic and can't go by the principles of one political ideology. Political ideologies are secular religions in this regard. Many self described liberals, libertarians, and conservatives rarely listen to each other on how to better society. Liberals see government as the only solution to all of society's ills. Conservatives and Libertarians find government as the mere deterrent to social ills and adhere to free market fundamentalism as holy. It's as if the free market makes everything a Utopia. These differences in dogma often resorts to divisive politics. How is that any different to religious differences? 

EXAMPLE OUTPUT:
{{
    "cogency": "Good",
    "effectiveness": "Good",
    "reasonableness": "Bad",
    "overall": "Bad"
}}

###EXAMPLE###
EXAMPLE argument:
Unless every single gun that is issued legally is tested with ballistics before being issued so that any bullet fired from a licensed gun can be traced, if found intact, then guns pose a threat because there is no way for those bullets to be traced back to owners so they can account for the shots they fired. It's like giving someone a jaguar as long as they promise to never let it out of their site. It's bull sh*t.

EXAMPLE OUTPUT:
{{
    "cogency": "Good",
    "effectiveness": "Bad",
    "reasonableness": "Good",
    "overall": "Good"
}}

###EXAMPLE###
EXAMPLE argument:
I've read some scientific racism writing and I think many of its claims have no scientific basis. Shouldn't the differences between races be studied ? 
People of different origins have different bodies (skeletton, skin color...) why wouldn't they have a different brain ?
Note that I don't think any race is superior to the other, just different. Just like the differences between gender doesn't make one superior to another.
On another hand I don't want such research to be done because that could be misinterpreted by hateful people and lead to a resurgence of racism.

EXAMPLE OUTPUT:
{{
    "cogency": "Bad",
    "effectiveness": "Bad",
    "reasonableness": "Good",
    "overall": "Bad"
}}
"""
example4 = """
###EXPECTED OUTPUT###

Respond ONLY with a JSON object. The values MUST be Effective or Ineffective:
{{
  "cogency": "Effective" | "Ineffective",
  "effectiveness": "Effective" | "Ineffective",
  "reasonableness": "Effective" | "Ineffective",
  "overall": "Effective" | "Ineffective"
}}

Always wrap all values inside double quotes, so the output is always valid JSON.

###EXAMPLE###
EXAMPLE argument:
Through cooperation, children can learn about interpersonal skills which are significant in the future life of all students.
What we acquired from team work is not only how to achieve the same goal with others but more importantly, how to get along with others.
During the process of cooperation, children can learn about how to listen to opinions of others, how to communicate with others, how to think comprehensively, and even how to compromise with other team members when conflicts occurred.
All of these skills help them to get on well with other people and will benefit them for the whole life.

EXAMPLE OUTPUT:
{{
    "cogency": "Effective",
    "effectiveness": "Ineffective",
    "reasonableness": "Effective",
    "overall": "Effective"
}}
"""
# --- Selector desde línea de comandos o input ---
print("\nSelect version of prompt (1 to 4):")
version = int(input("Enter version number: "))
print(f"\n✅ Using prompt version {version}...\n")

# --- Diccionarios para seleccionar automáticamente ---
common_intros = {
    1: common_intro1,
    2: common_intro2,
    3: common_intro3,
    4: common_intro4
}

examples = {
    1: example1,
    2: example2,
    3: example3,
    4: example4
}

# --- Prompt Builder según versión ---
def build_prompt(argument):
    selected_intro = common_intros.get(version, common_intro1)
    selected_example = examples.get(version, example1)
    return f"{selected_intro}\n{dimensions}\n{selected_example}\n\n###argument###\n{argument}###YOUR RESPONSE### (Only respond with the JSON object)"


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
        match = re.search(r'(\{{1,2})(.*?)(\}{1,2})', text, re.DOTALL)
        if not match:
            print("❌ No JSON-like object found in response. ")
            return None  
        
        json_text = match.group(0)

        if json_text.startswith('{{') and json_text.endswith('}}'):
            json_text = json_text[1:-1].strip()

        parsed = json.loads(json_text)

        expected_dims = ["cogency", "effectiveness", "reasonableness", "overall"]

        if all(dim in parsed for dim in expected_dims):
            if version == 1:
                # Valores numéricos
                return {dim: int(parsed[dim]) for dim in expected_dims}
            else:
                # Valores string (Good, Bad, Effective, etc.)
                return {dim: parsed[dim] for dim in expected_dims}
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
with open(output_filename, "w") as f:
    json.dump(all_runs, f, indent=2)

print(f"\n--- RESPUESTAS GUARDADAS EN: {output_filename} ---")
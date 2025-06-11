import requests
import  time
import json
from datetime import datetime
import os
import re

API_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "llama3.1"

essay = """<MajorClaim> The death penalty should be abolished because it is inhumane and ineffective as a deterrent.  
<Claim> Capital punishment violates the fundamental human right to life and dignity.  
<Premise> Numerous reports have documented cases where innocent people were executed due to flaws in the judicial system.  
<Premise> Executions are often carried out in a manner that causes prolonged suffering, which can be considered a form of torture.  
<Claim> The death penalty does not effectively reduce crime rates.  
<Premise> Studies have shown that regions with capital punishment do not have lower homicide rates than those without it.  
<Premise> Criminals often act in the heat of the moment or without expecting to be caught, so harsh punishments don’t deter them."""

common_intro = """
####ROLE###
You are an Argument Annotator AI.

###OBJECTIVE###
Given a segmented essay with labeled argument components (Major Claim, Claims, and Premises), assess the quality of each argument component.

###QUALITY RATINGS###
For each argument component (Major Claim, Claims, Premises), assign a quality rating of either "Effective" or "Ineffective" in each of the four categories.
"""

dimensions_v1 = """
###DIMENSIONS OF ARGUMENT QUALITY###

Assess each argument component based on the following dimensions. The overall rating (<Effective> or <Ineffective>) should be a reasoned synthesis of these aspects:

1. Logical Cogency:
   - Local Acceptability: Are the premises rationally credible to the intended audience?
   - Local Relevance: Does the component contribute meaningfully to supporting its conclusion?
   - Local Sufficiency: Do the premises, together, provide enough support to draw the conclusion?

2. Rhetorical Effectiveness:
   - Clarity: Is the argument expressed clearly and understandably?
   - Appropriateness: Is the tone suitable for the topic and audience?
   - Arrangement: Are ideas presented in a logically ordered structure?
   - Credibility: Does the component enhance the author’s reliability?
   - Emotional Appeal: Does it appropriately engage emotions to increase persuasion?

3. Dialectical Reasonableness:
   - Global Acceptability: Is the argument as a whole likely to be accepted by the audience?
   - Global Relevance: Does it contribute to resolving the broader issue?
   - Global Sufficiency: Does it address or pre-empt counterarguments adequately?

4. Overall Assessment: This is not a simple average of dimensions, but a holistic judgment. Components that perform well across most of the dimensions above should be rated <Effective>. Otherwise, rate them <Ineffective>.
"""

dimensions_v2 = """
###DIMENSIONS OF ARGUMENT QUALITY###

Each component should be evaluated holistically according to five dimensions. First, consider logical cogency: whether the component presents ideas that are credible, relevant to the claim or conclusion, and sufficient to justify it. Second, assess rhetorical effectiveness by looking at how clearly the idea is expressed, whether the tone fits the topic and audience, how well it is structured, whether it adds to the author’s credibility, and whether it uses emotional appeal appropriately. Third, examine dialectical reasonableness, or the extent to which the argument is acceptable to the audience, contributes to resolving the issue, and addresses possible counterarguments.  Finally, make an overall assessment: if the component performs well across most of these areas, label it <Effective>; if not, label it <Ineffective>.
"""

example = """
###EXPECTED OUTPUT###
- Format the output by starting with “#OUTPUT:” and ending with “#END.”
- Between those markers, return a JSON array, where each element represents one argument component
- Each element sohould be a JSON object containing the following keys:


{
"component": "<full text of the argument component>",
"type": "MajorClaim" | "Claim" | "Premise",
"cogency": "Effective" | "Ineffective",
"effectiveness": "Effective" | "Ineffective", 
"reasonableness": "Effective" | "Ineffective",
"overall": "Effective" | "Ineffective"
}


###EXAMPLE###
EXAMPLE ESSAY:
<Claim> through cooperation, children can learn about interpersonal skills which are significant in the future life of all students
<Premise> What we acquired from team work is not only how to achieve the same goal with others but more importantly, how to get along with others
<Premise> During the process of cooperation, children can learn about how to listen to opinions of others, how to communicate with others, how to think comprehensively, and even how to compromise with other team members when conflicts occurred
<Premise> All of these skills help them to get on well with other people and will benefit them for the whole life.

#OUTPUT:
[
  {
    "component": "through cooperation, children can learn about interpersonal skills which are significant in the future life of all students",
    "type": "Claim",
    "cogency": "Effective ",
    "effectiveness": "Effective",
    "reasonableness": "Ineffective",
    "overall": "Effective"
  },
  {
    "component": "What we acquired from team work is not only how to achieve the same goal with others but more importantly, how to get along with others",
    "type": "Premise",
    "cogency": "Ineffective",
    "effectiveness": "Ineffective",
    "reasonableness": "Ineffective",
    "overall": "Ineffective"
  },
  {
    "component": "During the process of cooperation, children can learn about how to listen to opinions of others, how to communicate with others, how to think comprehensively, and even how to compromise with other team members when conflicts occurred",
    "type": "Premise",
    "cogency": "Effective",
    "effectiveness": "Effective",
    "reasonableness": "Effective",
    "overall": "Effective"
  },
  {
    "component": "All of these skills help them to get on well with other people and will benefit them for the whole life",
    "type": "Premise",
    "cogency": "Ineffective",
    "effectiveness": "Effective",
    "reasonableness": "Ineffective",
    "overall": "Ineffective"
  }
]
#END.

"""

# This function builds the prompt based on the selected version
def build_prompt(version):
    dimensions = dimensions_v1 if version == "1" else dimensions_v2
    return f"{common_intro}\n{dimensions}\n{example}\n\n###ESSAY###\n{essay}"


# This function sends the prompt to the API and returns the response it also measures the response time
def send_prompt(prompt):
    start_time = time.time()
    response = requests.post(API_URL, json={
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False
    })
    end_time = time.time()
    elapsed = end_time - start_time
    if response.status_code != 200:
        raise Exception(f"Error {response.status_code}: {response.text}")
    
    print(f"Tiempo de respuesta: {elapsed:.3f} segundos")
    return response.json()["response"], elapsed

# This function extracts the JSON block from the response text
def extract_json_block(text):
    match = re.search(r'#OUTPUT:\s*(\[[\s\S]*?\])\s*#END', text, re.DOTALL)
    if not match:
        raise ValueError("No se encontró el bloque JSON en la respuesta.")
    return json.loads(match.group(1))

# this function saves the response to a file with a timestamp
def save_output_to_file(output,version, elapsed, folder_name):
    base_dir = os.path.join("responses", folder_name)
    os.makedirs(base_dir, exist_ok=True)  # Create the directory if it doesn't exist, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(base_dir, f"output_{version}_{timestamp}.json")
    # Add time as extra information
    data_to_save = {
        "time_seconds": round(elapsed, 3),
        "response": output
    }
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(data_to_save, f, ensure_ascii=False, indent=2)
    print(f"Respuesta guardada en {filename}")

# Main function to run the script
if __name__ == "__main__":
    print("Elige la versión del prompt que quieres probar:")
    print("1 - Versión detallada con subpuntos")
    print("2 - Versión resumida con síntesis por dimensión")
    version = input("Introduce 1 o 2: ").strip()
    if version not in ["1", "2"]:
        print("Versión inválida.")
    else:
        prompt = build_prompt(version)
        raw_response, elapsed = send_prompt(prompt)
        try:
            parsed_response = extract_json_block(raw_response)
            folder_name = input("Introduce el nombre de la carpeta para guardar la respuesta: ").strip()
            save_output_to_file(parsed_response, version, elapsed, folder_name)
            print("\n=== RESPUESTA DEL MODELO ===")
            print(json.dumps(parsed_response, indent=2, ensure_ascii=False))
        except Exception as e:
            print(f"Error al procesar la respuesta: {e}")

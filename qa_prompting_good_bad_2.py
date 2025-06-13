import requests
import  time
import json
from datetime import datetime
import os
import re

API_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "llama3.1"

essay = """I believe that this solves a number of issues that I have with the death penalty, amongst other benefits.
The state's hands are kept clean.
An innocent person will always choose the imprisonment, during this time new evidence may come to light that proves them innocent. No more dead innocents.
The convict's freedom of choice remains intact, which I consider to be important.
I believe that presenting them with a choice is more human than executing them against their will.
If they choose the life imprisonment, then they should be presented the option again every five years or so. This way the life imprisonment is not inhumane as there is always an \"out\".
The method of suicide should be left up to the convict.
No other people have to bear the burden of ending someone's life."""

common_intro = """
####ROLE###
You are an Argument Annotator AI.

###OBJECTIVE###
Given an argument, assess the quality.

###QUALITY RATINGS###
Assign a quality rating of either "Good" or "Bad" in each of the four categories. Internally, assign a value between 1-5 to each quality aspect. If the value is between 1-2, the quality is "Bad"; if it is between 3-5, the quality is "Good".
"""

dimensions_v1 = """
###DIMENSIONS OF ARGUMENT QUALITY###

Assess each argument component based on the following dimensions. The overall rating (<Good> or <Bad>) should be a reasoned synthesis of these aspects:

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

4. Overall Assessment: This is not a simple average of dimensions, but a holistic judgment. Components that perform well across most of the dimensions above should be rated <Good>. Otherwise, rate them <Bad>.
"""

dimensions_v2 = """
###DIMENSIONS OF ARGUMENT QUALITY###

The argument should be evaluated holistically according to four dimensions. First, consider logical cogency: whether the component presents ideas that are credible, relevant to the claim or conclusion, and sufficient to justify it. Second, assess rhetorical effectiveness by looking at how clearly the idea is expressed, whether the tone fits the topic and audience, how well it is structured, whether it adds to the author’s credibility, and whether it uses emotional appeal appropriately. Third, examine dialectical reasonableness, or the extent to which the argument is acceptable to the audience, contributes to resolving the issue, and addresses possible counterarguments. Finally, make an overall assessment: if the component performs well across most of these areas, label it <Good>; if not, label it <Neutral> or <Bad>.
"""

example = """
###EXPECTED OUTPUT###
- Format the output by starting with “#OUTPUT:” and ending with “#END.”
- Between those markers, return a JSON, containing the following keys:

{
"component": "<full text of the argument component>",
"cogency": "Good" | "Bad" ,
"effectiveness": "Good" | "Bad"  , 
"reasonableness": "Good" | "Bad" ,
"overall": "Good" | "Bad" 
}


###EXAMPLE###
EXAMPLE ESSAY:
Through cooperation, children can learn about interpersonal skills which are significant in the future life of all students
What we acquired from team work is not only how to achieve the same goal with others but more importantly, how to get along with others
During the process of cooperation, children can learn about how to listen to opinions of others, how to communicate with others, how to think comprehensively, and even how to compromise with other team members when conflicts occurred
All of these skills help them to get on well with other people and will benefit them for the whole life.

#OUTPUT:
[
  {
    "component": "Through cooperation, children can learn about interpersonal skills which are significant in the future life of all students. What we acquired from team work is not only how to achieve the same goal with others but more importantly, how to get along with others. During the process of cooperation, children can learn about how to listen to opinions of others, how to communicate with others, how to think comprehensively, and even how to compromise with other team members when conflicts occurred. All of these skills help them to get on well with other people and will benefit them for the whole life.",
    "cogency": "Good",
    "effectiveness": "Bad",
    "reasonableness": "Good",
    "overall": "Good"
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
    filename = os.path.join(base_dir, f"output_{version}_{timestamp}.txt")
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
        folder_name = input("Introduce el nombre de la carpeta para guardar la respuesta: ").strip()
        prompt = build_prompt(version)
        raw_response, elapsed = send_prompt(prompt)
        try:
            parsed_response = extract_json_block(raw_response)
            save_output_to_file(parsed_response, version, elapsed, folder_name)
            print("\n=== RESPUESTA DEL MODELO ===")
            print(json.dumps(parsed_response, indent=2, ensure_ascii=False))
        except Exception as e:
            print(f"Error al procesar la respuesta: {e}")

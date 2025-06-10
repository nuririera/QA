import requests

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
For each argument component (Major Claim, Claims, Premises), assign a quality rating of either <Effective> or <Ineffective>.
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

4. Deliberative Norms:
   - Rationality: Is the reasoning logically sound?
   - Interactivity: Does it acknowledge or engage with other perspectives?
   - Equality: Does it assume or promote equal participation?
   - Civility: Is it respectful and constructive in tone?
   - Orientation to the Common Good: Does it aim at communal rather than selfish outcomes?
   - Constructiveness: Does it seek consensus or resolution?
   - Alternative Communication: Does it incorporate or respect non-traditional forms (e.g., storytelling)?

5. Overall Assessment: This is not a simple average of dimensions, but a holistic judgment. Components that perform well across most of the dimensions above should be rated <Effective>. Otherwise, rate them <Ineffective>.
"""

dimensions_v2 = """
###DIMENSIONS OF ARGUMENT QUALITY###

Each component should be evaluated holistically according to five dimensions. First, consider logical cogency: whether the component presents ideas that are credible, relevant to the claim or conclusion, and sufficient to justify it. Second, assess rhetorical effectiveness by looking at how clearly the idea is expressed, whether the tone fits the topic and audience, how well it is structured, whether it adds to the author’s credibility, and whether it uses emotional appeal appropriately. Third, examine dialectical reasonableness, or the extent to which the argument is acceptable to the audience, contributes to resolving the issue, and addresses possible counterarguments. Fourth, take into account deliberative norms, such as whether the reasoning is logical, respectful, aimed at understanding others, oriented to the common good, and open to alternative ways of communicating ideas. Finally, make an overall assessment: if the component performs well across most of these areas, label it <Effective>; if not, label it <Ineffective>.
"""

example = """
###EXPECTED OUTPUT###
- Replicate the exact essay.
- At the beginning of each argument component, prepend its quality rating (either <Effective> or <Ineffective>).
- Format the output by starting with “#OUTPUT:” and ending with “#END.”

###EXAMPLE###
EXAMPLE ESSAY:
<Claim> through cooperation, children can learn about interpersonal skills which are significant in the future life of all students
<Premise> What we acquired from team work is not only how to achieve the same goal with others but more importantly, how to get along with others
<Premise> During the process of cooperation, children can learn about how to listen to opinions of others, how to communicate with others, how to think comprehensively, and even how to compromise with other team members when conflicts occurred
<Premise> All of these skills help them to get on well with other people and will benefit them for the whole life.

#OUTPUT:
<Effective> through cooperation, children can learn about interpersonal skills which are significant in the future life of all students  
<Ineffective> What we acquired from team work is not only how to achieve the same goal with others but more importantly, how to get along with others  
<Effective> During the process of cooperation, children can learn about how to listen to opinions of others, how to communicate with others, how to think comprehensively, and even how to compromise with other team members when conflicts occurred  
<Ineffective> All of these skills help them to get on well with other people and will benefit them for the whole life  
#END
"""

def build_prompt(version):
    dimensions = dimensions_v1 if version == "1" else dimensions_v2
    return f"{common_intro}\n{dimensions}\n{example}\n\n###ESSAY###\n{essay}"

def send_prompt(prompt):
    response = requests.post(API_URL, json={
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False
    })
    if response.status_code != 200:
        raise Exception(f"Error {response.status_code}: {response.text}")
    return response.json()["response"]

if __name__ == "__main__":
    print("Elige la versión del prompt que quieres probar:")
    print("1 - Versión detallada con subpuntos")
    print("2 - Versión resumida con síntesis por dimensión")
    version = input("Introduce 1 o 2: ").strip()
    if version not in ["1", "2"]:
        print("Versión inválida.")
    else:
        prompt = build_prompt(version)
        output = send_prompt(prompt)
        print("\n=== RESPUESTA DEL MODELO ===")
        print(output)

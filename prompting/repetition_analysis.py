import requests
import time
import json
import os
import re
from statistics import mean, variance
from collections import defaultdict
from datetime import datetime

API_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "llama3.1"
REPEATS = 25

# Define el ensayo fijo
essay = """<MajorClaim> The death penalty should be abolished because it is inhumane and ineffective as a deterrent.  
<Claim> Capital punishment violates the fundamental human right to life and dignity.  
<Premise> Numerous reports have documented cases where innocent people were executed due to flaws in the judicial system.  
<Premise> Executions are often carried out in a manner that causes prolonged suffering, which can be considered a form of torture.  
<Claim> The death penalty does not effectively reduce crime rates.  
<Premise> Studies have shown that regions with capital punishment do not have lower homicide rates than those without it.  
<Premise> Criminals often act in the heat of the moment or without expecting to be caught, so harsh punishments don’t deter them."""

# Función para extraer sección por versión del archivo prompt_template.txt
def extract_prompt_section(template_text, version):
    pattern = None
    if version == "1":
        # Extrae sección entre ###VERSION_1_START### y ###VERSION_1_END###
        pattern = r"###VERSION_1_START###([\s\S]*?)###VERSION_1_END###"
    elif version == "2":
        # Extrae sección entre ###VERSION_2_START### y ###VERSION_2_END###
        pattern = r"###VERSION_2_START###([\s\S]*?)###VERSION_2_END###"
    else:
        raise ValueError("Versión inválida")

    match = re.search(pattern, template_text)
    if not match:
        raise ValueError(f"No se encontró la sección para la versión {version} en el archivo de plantilla.")
    return match.group(1).strip()

# Enviar petición
def send_prompt(prompt):
    response = requests.post(API_URL, json={
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False
    })
    if response.status_code != 200:
        raise Exception(f"Error {response.status_code}: {response.text}")
    return response.json()["response"]

# Extraer JSON del bloque
def extract_json_block(text):
    match = re.search(r'#OUTPUT:\s*(\[[\s\S]*?\])\s*#END', text, re.DOTALL)
    if not match:
        raise ValueError("No se encontró el bloque JSON en la respuesta.")
    return json.loads(match.group(1))

# Convertir valor textual a numérico
def binarize(value):
    return 1 if value.strip().lower() == "effective" else 0

# Acumular datos y calcular estadísticas
def analyze_responses(responses):
    data = defaultdict(lambda: defaultdict(list))

    for response in responses:
        for entry in response:
            component = entry["component"].strip()
            for dimension in ["cogency", "effectiveness", "reasonableness", "overall"]:
                score = binarize(entry[dimension])
                data[component][dimension].append(score)

    stats = defaultdict(dict)
    for comp, dims in data.items():
        for dim, values in dims.items():
            stats[comp][dim] = {
                "mean": round(mean(values), 3),
                "variance": round(variance(values), 3) if len(values) > 1 else 0.0,
                "n": len(values)
            }
    return stats

# Guardar resultados
def save_results(responses, stats):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    folder = f"repetition_results_{timestamp}"
    os.makedirs(folder, exist_ok=True)

    with open(os.path.join(folder, "raw_responses.json"), "w", encoding="utf-8") as f:
        json.dump(responses, f, indent=2, ensure_ascii=False)

    with open(os.path.join(folder, "statistics.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    print(f"Guardado en: {folder}")

# MAIN
if __name__ == "__main__":
    version = input("Selecciona la versión del prompt a usar (1 o 2): ").strip()
    if version not in ["1", "2"]:
        print("Versión inválida.")
        exit(1)

    with open("prompt_template.txt", "r", encoding="utf-8") as f:
        template_text = f.read()

    try:
        prompt_section = extract_prompt_section(template_text, version)
    except Exception as e:
        print(f"Error al extraer prompt: {e}")
        exit(1)

    prompt = f"{prompt_section}\n\n###ESSAY###\n{essay}"

    print(f"Lanzando {REPEATS} repeticiones con la versión {version}...")
    all_responses = []
    for i in range(REPEATS):
        print(f"→ Ejecución {i+1}/{REPEATS}...")
        try:
            raw = send_prompt(prompt)
            parsed = extract_json_block(raw)
            all_responses.append(parsed)
        except Exception as e:
            print(f"Error en ejecución {i+1}: {e}")

    print("Calculando estadísticas...")
    stats = analyze_responses(all_responses)
    save_results(all_responses, stats)
    print("✔ Análisis completado.")

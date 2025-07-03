import datetime
import sys
from Logger import Logger
import time
from dataset_division import test_data
import json
from analyze_results_not_binary import evaluate_single_run, analyze_variability_across_runs, evaluate_multiple_runs
import os


log_dir = "evaluation"
os.makedirs(log_dir, exist_ok=True)
date = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M")
log_filename = os.path.join(log_dir, f"evaluation_{date}.txt")
sys.stdout = Logger(log_filename)

global_start = time.time()

# Pedir esquema al usuario por terminal
print("Available schemas:")
print("1: binary_good_bad")
print("2: ternary_bad_medium_good")
print("3: binary_effective_ineffective")
print("4: numeric_1_to_5")
schema_option = input("Select the label schema (1-4): ").strip()

schemas_map = {
    "1": "binary_good_bad",
    "2": "ternary_bad_medium_good",
    "3": "binary_effective_ineffective",
    "4": "numeric_1_to_5",
}

if schema_option not in schemas_map:
    print("Invalid option. Finishing execution.")
    sys.exit(1)

schema_name = schemas_map[schema_option]
print(f"Using schema: {schema_name}")

# Buscar archivos de respuesta generados por el modelo
response_dir = "model_responses"
response_files = [f for f in os.listdir(response_dir) if f.startswith("model_responses_") and f.endswith(".json")]

if not response_files:
    print("No model response files found in the 'model_responses' directory.")
    sys.exit(1)

# Mostrar opciones al usuario
print("Response files available:")
for idx, file in enumerate(response_files):
    print(f"{idx + 1}: {file}")

# Solicitar selección
try:
    selected_index = int(input("Select a file by number to analyze: ")) - 1
except ValueError:
    print("Invalid input. Please enter a number.")
    sys.exit(1)

# Validar selección
if selected_index < 0 or selected_index >= len(response_files):
    print("Invalid selection.")
    sys.exit(1)

# Cargar el archivo seleccionado
input_filename = response_files[selected_index]
with open(input_filename, "r") as f:
    all_runs = json.load(f)

# Obtener etiquetas del ground truth
ground_truth = [entry["labels"] for entry in test_data]

print(f"\n=== Number of runs: {len(all_runs)} ===")
print(f"=== Number of arguments per run: {len(all_runs[0]) if all_runs else 0} ===")

# Ahora pasamos schema_name a las funciones para que usen el esquema correcto

# Evaluar cada ejecución contra el ground truth
for i, run in enumerate(all_runs):
    print(f"\n--- EVALUATING RUN {i + 1} ---")
    evaluate_single_run(run, ground_truth, schema_name)

# Análisis de agregación en múltiples ejecuciones
print("\n--- ANALYSIS OF AGGREGATION IN MULTIPLE RUNS ---")
evaluate_multiple_runs(all_runs, ground_truth, schema_name)

# Análisis de variabilidad entre ejecuciones
print("\n--- ANALYSIS OF VARIABILITY BETWEEN RUNS (ACROSS RUNS) ---")
analyze_variability_across_runs(all_runs, ground_truth, schema_name)

# Mostrar tiempo total
total_duration = time.time() - global_start
print(f"\n---TOTAL EXECUTION TIME ---")
print(f"Total time: {total_duration:.2f} seconds")

import datetime
import sys
from Logger import Logger
import time
from dataset_division import test_data
import json
from analyze_results_not_binary import evaluate_single_run, analyze_variability_across_runs, evaluate_multiple_runs
import os

# Obtener fecha y hora actual en formato YYYY-MM-DD-HH-MM
date = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M")

# Nombre del archivo de log
log_filename = f"evaluacion_{date}.txt"

# Redirigir la salida estándar al Logger
sys.stdout = Logger(log_filename)

global_start = time.time()  # Medir tiempo total de ejecución

# Pedir esquema al usuario por terminal
print("Esquemas disponibles:")
print("1: binary_good_bad")
print("2: ternary_bad_medium_good")
print("3: binary_effective_ineffective")
print("4: numeric_1_to_5")
schema_option = input("Selecciona el esquema de etiquetas (1-4): ").strip()

schemas_map = {
    "1": "binary_good_bad",
    "2": "ternary_bad_medium_good",
    "3": "binary_effective_ineffective",
    "4": "numeric_1_to_5",
}

if schema_option not in schemas_map:
    print("Opción inválida. Terminando ejecución.")
    sys.exit(1)

schema_name = schemas_map[schema_option]
print(f"Usando esquema: {schema_name}")

# Buscar archivos de respuesta generados por el modelo
response_files = [f for f in os.listdir() if f.startswith("model_responses_") and f.endswith(".json")]

if not response_files:
    print("No se encontraron archivos tipo model_responses_*.json en el directorio actual.")
    sys.exit(1)

# Mostrar opciones al usuario
print("Archivos de respuesta de modelo disponibles:")
for idx, file in enumerate(response_files):
    print(f"{idx + 1}: {file}")

# Solicitar selección
try:
    selected_index = int(input("Selecciona el número del archivo que quieres evaluar: ")) - 1
except ValueError:
    print("Entrada inválida. Por favor, introduce un número.")
    sys.exit(1)

# Validar selección
if selected_index < 0 or selected_index >= len(response_files):
    print("Selección no válida.")
    sys.exit(1)

# Cargar el archivo seleccionado
input_filename = response_files[selected_index]
with open(input_filename, "r") as f:
    all_runs = json.load(f)

# Obtener etiquetas del ground truth
ground_truth = [entry["labels"] for entry in test_data]

print(f"\n=== Número de ejecuciones cargadas: {len(all_runs)} ===")
print(f"=== Número de argumentos por ejecución: {len(all_runs[0]) if all_runs else 0} ===")

# Ahora pasamos schema_name a las funciones para que usen el esquema correcto

# Evaluar cada ejecución contra el ground truth
for i, run in enumerate(all_runs):
    print(f"\n--- EVALUACIÓN DE LA EJECUCIÓN {i + 1} ---")
    evaluate_single_run(run, ground_truth, schema_name)

# Análisis de agregación en múltiples ejecuciones
print("\n--- ANÁLISIS DE VARIABILIDAD ENTRE ARGUMENTOS (ACROSS ARGUMENTS) ---")
evaluate_multiple_runs(all_runs, ground_truth, schema_name)

# Análisis de variabilidad entre ejecuciones
print("\n--- ANÁLISIS DE VARIABILIDAD ENTRE EJECUCIONES (ACROSS RUNS) ---")
analyze_variability_across_runs(all_runs, ground_truth, schema_name)

# Mostrar tiempo total
total_duration = time.time() - global_start
print(f"\n--- TIEMPO TOTAL DE EJECUCIÓN DE LA EVALUACIÓN ---")
print(f"Tiempo total: {total_duration:.2f} segundos")

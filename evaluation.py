import datetime
import sys
from Logger import Logger
import time
from dataset_division import test_data
import json 
from analyze_results import evaluate_single_run, analyze_variability_across_runs, evaluate_multiple_runs
import os 

#Date in YYYY-MM-DD-HH-MM format
date = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M")

#Name of the file
log_filename = f"evaluacion_{date}.txt"

sys.stdout = Logger(log_filename)
global_start = time.time() #total time

# List all JSON files starting with 'model_responses_'
response_files = [f for f in os.listdir() if f.startswith("model_responses_") and f.endswith(".json")]

# If no files found, exit early
if not response_files:
    print("No model_responses_*.json files found in current directory.")
    sys.exit(1)

# Show the available options
print("Available model response files:")
for idx, file in enumerate(response_files):
    print(f"{idx + 1}: {file}")

# Ask user to select one
selected_index = int(input("Select a file number to evaluate: ")) - 1

# Validate selection
if selected_index < 0 or selected_index >= len(response_files):
    print("Invalid selection.")
    sys.exit(1)

# Use the selected file
input_filename = response_files[selected_index]

with open(input_filename, "r") as f:
    all_runs = json.load(f)

# Ground truth labels
ground_truth = [entry["labels"] for entry in test_data]

print(f"=== Number of runs loaded: {len(all_runs)} ===")
print(f"=== Number of arguments per run: {len(all_runs[0]) if all_runs else 0} ===")

# Evaluate each run against the ground truth
for i, run in enumerate(all_runs):
    print(f"\n--- EVALUATION OF RUN {i + 1} ---")
    evaluate_single_run(run, ground_truth)

# Analyze variability across arguments
print("\n--- VARIABILITY ANALYSIS ACROSS ARGUMENTS ---")
evaluate_multiple_runs(all_runs, ground_truth)

# Analyze variability across runs
print("\n--- VARIABILITY ANALYSIS ACROSS RUNS ---")
analyze_variability_across_runs(all_runs)

total_duration = time.time() - global_start
print(f"\n--- TOTAL EVALUATION EXECUTION TIME ---")
print(f"Total time: {total_duration:.2f} seconds")
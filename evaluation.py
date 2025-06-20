import datetime
import sys
from Logger import Logger
import time
from dataset_division import test_data
import json 
from analyze_results import evaluate_single_run, analyze_variability_across_runs, evaluate_multiple_runs

#Date in YYYY-MM-DD-HH-MM format
date = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M")

#Name of the file
log_filename = f"evaluacion_{date}.txt"

sys.stdout = Logger(log_filename)
global_start = time.time() #total time

# Path to the JSON file with the saved model responses
input_filename = "model_responses_2025-06-20-12-04.json"

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
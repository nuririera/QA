import numpy as np
from statistics import mean
from sklearn.metrics import (f1_score, confusion_matrix, classification_report, precision_recall_fscore_support)

# Define the mapping for ratings
# The ratings are "Good" and "Bad", we map them to 1 and 0 respectively
rating_map = {"Good": 1, "Bad": 0}
dimensions = ["cogency", "effectiveness", "reasonableness", "overall"]


# Calculate basic metrics and reports for a set of labels
# returns a dictionary with the metrics
def compute_metrics(true_scores, model_scores):
    f1 = f1_score(true_scores, model_scores)
    precision, recall, fscore, _ = precision_recall_fscore_support(true_scores, model_scores, zero_division=0, average='binary')

    cm = confusion_matrix(true_scores, model_scores)
    report = classification_report(true_scores, model_scores, zero_division=0)
    return {
        "f1": f1,
        "precision": precision,
        "recall": recall, 
        "fscore": fscore, 
        "cm": cm,
        "classification_report": report
    }

# Function to analyze the results of the model outputs against ground truths
def evaluate_single_run(model_outputs, ground_truths):
    print("\n --- ANALYSIS RESULTS --- (evaluating model outputs against ground truths - 1 run)\n")
    # List of dimensions to analyze

    for dim in dimensions:
        # Turn the dimension into a key for accessing the ratings
        model_scores = [rating_map[x[dim]] for x in model_outputs]
        true_scores = [rating_map[x[dim]] for x in ground_truths]

        metrics = compute_metrics(true_scores, model_scores)

        print(f"\n --- {dim.upper()} ---")
        # Metric of basic classification performance
        print(f"F1 Score: {metrics['f1']:.2f}")
        print("\nConfusion Matrix (Actual vs Predicted):")
        print("               Predicted")
        print("               Bad     Good")
        cm = metrics["cm"]
        print(f"True Bad   [{cm[0][0]:<5}  {cm[0][1]:<5}]")
        print(f"True Good  [{cm[1][0]:<5}  {cm[1][1]:<5}]")
        print("Classification Report:")
        print(metrics["classification_report"])

def evaluate_multiple_runs(model_outputs_runs, ground_truths):
    #model_outputs_runs: list of list of dicts -> [run1_outputs, run2_outputs, ...]
    #ground_truths: list of dicts -> [ground_truth1, ground_truth2, ...]

    print("\n === AGGREGATED ANALYSIS ACROSS MULTIPLE RUNS ===\n")
    n_runs = len(model_outputs_runs)

    # metrics storage: {"f1": [...], "precision": [...], "recall": [...]}
    metrics_by_dim = {dim: {"f1":[], "precision":[], "recall":[]} for dim in dimensions}

    for model_outputs in model_outputs_runs:
        for dim in dimensions:
            model_scores = [rating_map[x[dim]] for x in model_outputs]
            true_scores = [rating_map[x[dim]] for x in ground_truths]

            metrics = compute_metrics(true_scores, model_scores)

            metrics_by_dim[dim]["f1"].append(metrics["f1"])
            metrics_by_dim[dim]["precision"].append(metrics["precision"])
            metrics_by_dim[dim]["recall"].append(metrics["recall"])

    print(f"\n === AVERAGE METRICS ACROSS {n_runs} RUNS ===")
    for dim in dimensions:
        avg_f1 = mean(metrics_by_dim[dim]["f1"])
        avg_precision = mean(metrics_by_dim[dim]["precision"])
        avg_recall = mean(metrics_by_dim[dim]["recall"])

        print(f"\n --- {dim.upper()} ---")
        print(f"Average F1 Score: {avg_f1:.2f}")
        print(f"Average Precision: {avg_precision:.2f}")
        print(f"Average Recall: {avg_recall:.2f}")

def analyze_variability_across_runs(runs_outputs):
    print("\n --- ANALYSIS RESULTS --- (analyzing variability across multiple runs)\n")
    

    n_runs = len(runs_outputs)
    n_args = len(runs_outputs[0])

    for dim in dimensions:
        print(f"\n --- VARIABILITY IN {dim.upper()} ---")
        # bin_matrix[i][j] = binary evaluation of argument i in run j
        bin_matrix = np.array([
            [rating_map[run[i][dim]] for run in runs_outputs]
            for i in range(n_args)
        ])

        # Variance per argument: measures how much the evaluations vary across runs for that argument
        var_by_argument = np.var(bin_matrix, axis=1)
        mean_var = mean(var_by_argument)

        # Proporción de argumentos con desacuerdo entre runs
        desacuerdo = 1 - np.mean([
            len(set(row)) == 1 for row in bin_matrix
        ])

        print(f"Mean variance per argument:  {mean_var:.2f}")
        print(f"Proportion of arguments with disagreement across runs: {desacuerdo:.2%}")

        print("(filas = argumentos, columnas = runs):")
        print(bin_matrix)

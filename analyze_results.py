import numpy as np
from statistics import mean
from sklearn.metrics import (f1_score, confusion_matrix, classification_report, precision_recall_fscore_support)

# Define the mapping for ratings
# The ratings are "Good" and "Bad", we map them to 1 and 0 respectively
rating_map = {"Good": 1, "Bad": 0}
dimensions = ["cogency", "effectiveness", "reasonableness", "overall"]

# Binarization of the ratings
def binarize_scores(data, dim):
    return [rating_map[x[dim]] for x in data]


# Print a confusion matrix
def print_cm(cm):
    print("\nConfusion Matrix (Actual vs Predicted):")
    print("               Predicted")
    print("               Bad     Good")
    print(f"True Bad   [{cm[0][0]:<5.2f}  {cm[0][1]:<5.2f}]")
    print(f"True Good  [{cm[1][0]:<5.2f}  {cm[1][1]:<5.2f}]")

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
    print("\n --- ANALYSIS RESULTS --- (evaluating model outputs against ground truths - SINGLE RUN)\n")
    # List of dimensions to analyze

    for dim in dimensions:
        # Turn the dimension into a key for accessing the ratings
        model_scores = binarize_scores(model_outputs, dim)
        true_scores = binarize_scores(ground_truths, dim)

        metrics = compute_metrics(true_scores, model_scores)

        print(f"\n --- {dim.upper()} ---")
        # Metric of basic classification performance
        cm = metrics["cm"]
        print_cm(cm)
        print("Classification Report:")
        print(metrics["classification_report"])

# Compute de average confusion matrix across multiple runs by dimension
def compute_avg_cm(model_outputs_runs, ground_truths):
    n_runs = len(model_outputs_runs)
    avg_cms = {}

    for dim in dimensions:
        total_cm = np.zeros((2, 2), dtype=float)
        for run_outputs in model_outputs_runs:
            model_scores = binarize_scores(run_outputs, dim)
            true_scores = binarize_scores(ground_truths, dim)
            cm = confusion_matrix(true_scores, model_scores)
            total_cm += cm
        
        avg_cms[dim] = total_cm / n_runs

    return avg_cms

# Compute avg classification report across multiple runs by dimension
def compute_avg_report(model_outputs_runs, ground_truths):
    from collections import defaultdict

    n_runs = len(model_outputs_runs)
    avg_reports = {}

    for dim in dimensions:
        # Inicializar estructura para acumular
        total_report = defaultdict(lambda: defaultdict(float))

        for run_outputs in model_outputs_runs:
            model_scores = binarize_scores(run_outputs, dim)
            true_scores = binarize_scores(ground_truths, dim)
            report = classification_report(true_scores, model_scores, zero_division=0, output_dict=True)

            for label, metrics in report.items():
                if isinstance(metrics, dict):
                    for metric_name, value in metrics.items():
                        total_report[label][metric_name] += value
                else:
                    total_report[label] += metrics
        
        # Average
        averaged = {}
        for label, metrics in total_report.items():
            if isinstance(metrics, dict):
                averaged[label] = {k: v / n_runs for k, v in metrics.items()}
            else:
                averaged[label] = metrics / n_runs
           
        avg_reports[dim] = averaged
    
    return avg_reports


def evaluate_multiple_runs(model_outputs_runs, ground_truths):
    #model_outputs_runs: list of list of dicts -> [run1_outputs, run2_outputs, ...]
    #ground_truths: list of dicts -> [ground_truth1, ground_truth2, ...]

    print("\n === AGGREGATED ANALYSIS ACROSS MULTIPLE RUNS ===\n")
    n_runs = len(model_outputs_runs)

    metrics_by_dim = {dim: {"f1":[], "precision":[], "recall":[]} for dim in dimensions}

    for model_outputs in model_outputs_runs:
        for dim in dimensions:
            model_scores = binarize_scores(model_outputs, dim)
            true_scores = binarize_scores(ground_truths, dim)
            metrics = compute_metrics(true_scores, model_scores)

            for metric in ["f1", "precision", "recall"]:
                metrics_by_dim[dim][metric].append(metrics[metric])

    print(f"\n === AVERAGE METRICS ACROSS {n_runs} RUNS ===")
    for dim in dimensions:
        print(f"\n --- {dim.upper()} ---")
        for metric in ["f1", "precision", "recall"]:
            avg = mean(metrics_by_dim[dim][metric])
            print(f"Average {metric.upper()}: {avg:.2f}")

    print("\n === CONFUSION MATRICES ACROSS RUNS ===")
    avg_cms = compute_avg_cm(model_outputs_runs, ground_truths)
    for dim in dimensions:
        print(f"\n --- {dim.upper()} ---")
        print_cm(avg_cms[dim])

    print("\n === CLASSIFICATION REPORT ACROSS RUNS ===")
    avg_reports = compute_avg_report(model_outputs_runs, ground_truths)
    for dim in dimensions:
        print(f"\n --- {dim.upper()} ---")
        for label, metrics in avg_reports[dim].items():
            if isinstance(metrics, dict):
                print(f"Class {label}:")
                for metric_name, value in metrics.items():
                    print(f"  {metric_name}: {value:.2f}")
            else:
                print(f"{label}: {metrics:.2f}")

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

        print("(rows = arguments, columns = runs):")
        print(bin_matrix)

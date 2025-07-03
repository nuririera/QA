import datetime
import sys
import json
import os
import numpy as np
from scipy.stats import pearsonr
from sklearn.metrics import confusion_matrix, classification_report
from collections import defaultdict
from dataset_division import test_data
from Logger import Logger

dimensions = ["cogency", "effectiveness", "reasonableness", "overall"]

log_dir = "evaluation"
os.makedirs(log_dir, exist_ok=True)
date = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M")
log_filename = os.path.join(log_dir, f"evaluation_{date}.txt")
sys.stdout = Logger(log_filename)

def normalize_for_dimension(value, dimension_name=None):
    if value is None:
        return None
    try:
        if isinstance(value, str):
            val_lower = value.lower()
            if val_lower == "good":
                return 1
            elif val_lower == "bad":
                return 0
            else:
                fv = float(value)
        else:
            fv = float(value)

        threshold = 3 if dimension_name == "reasonableness" else 3.3
        if fv < threshold:
            return 0  # Bad
        else:
            return 1  # Good
    
    except Exception as e:
        print(f"Warning: Could not parse value '{value}' for dimension '{dimension_name}' - Error: {e}")
        return None

def prepare_scores(data, dim):
    scores = []
    for x in data:
        val = normalize_for_dimension(x[dim], dim)
        if val is not None:
            scores.append(val)
        else:
            print(f"Warning: '{x[dim]}' not valid for dimension '{dim}'")
            scores.append(-1)
    return scores

def print_dynamic_cm(cm, labels):
    print("\nConfusion Matrix (Actual vs Predicted):")
    if cm.shape[0] != len(labels):
        print(f"Warning: mismatch cm shape {cm.shape} and labels {len(labels)}")
        return
    header = " ".join(f"{l:>8}" for l in labels)
    print(f"{'':10}{header}")
    for i, row in enumerate(cm):
        row_str = " ".join(f"{val:8.2f}" for val in row)
        print(f"{labels[i]:10}{row_str}")

def evaluate_single_run(model_outputs, ground_truths):
    print("\n--- SINGLE RUN EVALUATION ---\n")
    for dim in dimensions:
        true_scores = prepare_scores(ground_truths, dim)
        model_scores = prepare_scores(model_outputs, dim)

        paired = [(t, p) for t, p in zip(true_scores, model_scores) if t >= 0 and p >= 0]
        if not paired:
            print(f"No valid samples for {dim}, skipping.")
            continue

        y_true = [t for t, _ in paired]
        y_pred = [p for _, p in paired]

        classes = sorted(set(y_true + y_pred))
        cm = confusion_matrix(y_true, y_pred, labels=classes)
        report = classification_report(y_true, y_pred, labels=classes, zero_division=0)

        print(f"\n--- {dim.upper()} ---")
        print_dynamic_cm(cm, [str(c) for c in classes])
        print("\nClassification Report:")
        print(report)

        # Calcular correlación Pearson
        corr, pval = pearsonr(y_true, y_pred)
        print(f"Pearson correlation for {dim}: {corr:.4f} (p={pval:.4f})")

def analyze_variability_and_correlation_across_runs(all_runs, ground_truths):
    print("\n--- VARIABILITY AND CORRELATION ACROSS RUNS ---\n")

    n_args = len(all_runs[0])
    n_runs = len(all_runs)

    for dim in dimensions:
        print(f"\n--- DIMENSION: {dim.upper()} ---")

        gt_scores = prepare_scores(ground_truths, dim)

        # Matriz (args x runs) con predicciones normalizadas
        preds_matrix = np.array([
            [normalize_for_dimension(all_runs[run_idx][arg_idx][dim], dim) for run_idx in range(n_runs)]
            for arg_idx in range(n_args)
        ])

        std_by_arg = np.std(preds_matrix, axis=1)
        mean_std = np.mean(std_by_arg)

        # Discrepancias
        disagreement_flags = [len(set(preds_matrix[i])) > 1 for i in range(n_args)]
        num_disagreements = sum(disagreement_flags)
        proportion_disagreement = num_disagreements / n_args

        print(f"Mean std deviation per argument: {mean_std:.4f}")
        print(f"Arguments with disagreement: {num_disagreements} / {n_args} ({proportion_disagreement:.2%})")

        # Correlación Pearson para cada run vs GT
        print("\nPearson correlation per run:")
        for run_idx in range(n_runs):
            run_preds = preds_matrix[:, run_idx]
            valid_mask = (run_preds >= 0) & (np.array(gt_scores) >= 0)
            if np.sum(valid_mask) == 0:
                print(f" Run {run_idx+1}: No valid data")
                continue
            corr, pval = pearsonr(np.array(gt_scores)[valid_mask], run_preds[valid_mask])
            print(f" Run {run_idx+1}: correlation = {corr:.4f} (p={pval:.4f})")

        # Correlación general: media de predicciones por argumento vs GT
        mean_preds = np.mean(preds_matrix, axis=1)
        valid_mask = (mean_preds >= 0) & (np.array(gt_scores) >= 0)
        if np.sum(valid_mask) > 0:
            corr, pval = pearsonr(np.array(gt_scores)[valid_mask], mean_preds[valid_mask])
            print(f"\nOverall Pearson correlation (mean across runs): {corr:.4f} (p={pval:.4f})")
        else:
            print("\nNo valid data for overall Pearson correlation.")

def compute_avg_cm_and_std(all_runs, ground_truth):
    n_runs = len(all_runs)
    avg_cms = {}

    for dim in dimensions:
        all_classes = set()
        true_scores_total = prepare_scores(ground_truth, dim)
        for run in all_runs:
            model_scores_total = prepare_scores(run, dim)
            paired = [(t, p) for t, p in zip(true_scores_total, model_scores_total) if t >= 0 and p >= 0]
            for t, p in paired:
                all_classes.update([t, p])
        all_classes = sorted(all_classes)

        cm_stack = []
        for run in all_runs:
            true_scores = prepare_scores(ground_truth, dim)
            model_scores = prepare_scores(run, dim)
            paired = [(t, p) for t, p in zip(true_scores, model_scores) if t >= 0 and p >= 0]
            if not paired:
                continue
            y_true = [t for t, _ in paired]
            y_pred = [p for _, p in paired]
            cm = confusion_matrix(y_true, y_pred, labels=all_classes)
            cm_stack.append(cm)

        if cm_stack:
            cm_array = np.stack(cm_stack)  # shape: (n_runs, n_classes, n_classes)
            mean_cm = np.mean(cm_array, axis=0)
            std_cm = np.std(cm_array, axis=0)
            avg_cms[dim] = (mean_cm, std_cm, all_classes)
        else:
            avg_cms[dim] = (None, None, [])

    return avg_cms

def print_avg_cm(avg_cms):
    for dim in dimensions:
        mean_cm, std_cm, labels = avg_cms[dim]
        print(f"\n--- Average Confusion Matrix for {dim.upper()} ---")
        if mean_cm is None:
            print("No data to display.")
            continue

        print("Mean Confusion Matrix:")
        print_dynamic_cm(mean_cm, [str(l) for l in labels])
        print("\nStd Dev Confusion Matrix:")
        print_dynamic_cm(std_cm, [str(l) for l in labels])

def compute_avg_classification_report(all_runs, ground_truth):
    n_runs = len(all_runs)
    avg_reports = {}

    for dim in dimensions:
        total_report = defaultdict(lambda: defaultdict(float))
        classes = None

        for run in all_runs:
            true_scores = prepare_scores(ground_truth, dim)
            model_scores = prepare_scores(run, dim)
            paired = [(t, p) for t, p in zip(true_scores, model_scores) if t >= 0 and p >= 0]
            if not paired:
                continue
            y_true = [t for t, _ in paired]
            y_pred = [p for _, p in paired]
            run_classes = sorted(set(y_true + y_pred))
            if classes is None:
                classes = run_classes

            report = classification_report(y_true, y_pred, labels=run_classes, output_dict=True, zero_division=0)
            for label, metrics in report.items():
                if isinstance(metrics, dict):
                    for metric_name, value in metrics.items():
                        total_report[label][metric_name] += value

        # Promediar
        averaged = {}
        for label, metrics in total_report.items():
            averaged[label] = {k: v / n_runs for k, v in metrics.items()}
        avg_reports[dim] = (averaged, classes)

    return avg_reports

def print_avg_classification_report(avg_reports):
    for dim in dimensions:
        print(f"\n--- Average Classification Report for {dim.upper()} ---")
        report, classes = avg_reports[dim]
        if not report:
            print("No data to display.")
            continue
        print(f"{'Label':<15} {'Precision':>9} {'Recall':>7} {'F1-score':>9} {'Support':>8}")
        print("-" * 50)
        for label in report:
            metrics = report[label]
            if isinstance(metrics, dict):
                print(f"{label:<15} {metrics.get('precision', 0):9.2f} {metrics.get('recall', 0):7.2f} {metrics.get('f1-score', 0):9.2f} {metrics.get('support', 0):8.0f}")



# --- Main ---

print(f"Loaded {len(test_data)} test arguments")

response_dir = "model_responses"
response_files = [f for f in os.listdir(response_dir) if f.startswith("model_responses_") and f.endswith(".json")]
if not response_files:
    print("No response files found.")
    sys.exit(1)

print("Available model response files:")
for i, f in enumerate(response_files):
    print(f" {i+1}: {f}")

selected_idx = int(input("Select response file number: ")) - 1
if selected_idx < 0 or selected_idx >= len(response_files):
    print("Invalid selection.")
    sys.exit(1)

with open(response_files[selected_idx], "r") as f:
    all_runs = json.load(f)

ground_truth = [entry["labels"] for entry in test_data]

print(f"\nLoaded {len(all_runs)} runs with {len(all_runs[0])} arguments each.")

# Evaluar cada ejecución individual
for i, run in enumerate(all_runs):
    print(f"\n=== Evaluation for run {i+1} ===")
    evaluate_single_run(run, ground_truth)

# Análisis y correlaciones across runs
analyze_variability_and_correlation_across_runs(all_runs, ground_truth)

avg_cms = compute_avg_cm_and_std(all_runs, ground_truth)
print_avg_cm(avg_cms)

avg_reports = compute_avg_classification_report(all_runs, ground_truth)
print_avg_classification_report(avg_reports)


print("\n--- Evaluation finished ---")

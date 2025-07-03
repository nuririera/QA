import numpy as np
from statistics import mean
from sklearn.metrics import confusion_matrix, classification_report
from collections import defaultdict

dimensions = ["cogency", "effectiveness", "reasonableness", "overall"]

# normalize a value for a specific dimension based on the schema
def normalize_for_dimension(value, schema_name, dimension_name=None):
    val = str(value).strip()

    if schema_name == "numeric_1_to_5":
        # Numeric values from 1 to 5
        try:
            iv = int(val)
            if 1 <= iv <= 5:
                return iv
            else:
                return None
        except:
            return None

    elif schema_name == "binary_good_bad":

        if val is None:
            return None

        try:
            # if its a string, check for "good" or "bad"
            if isinstance(val, str):
                if val.lower() == "good":
                    return 1
                elif val.lower() == "bad":
                    return 0
                else:
                    fv = float(val)
            else:
                fv = float(val)

            threshold = 3 if dimension_name == "reasonableness" else 3.3
            # Apply threshold logic
            if fv < threshold:
                return 0  # Bad
            else:
                return 1  # Good

        except Exception as e:
            print(f"Warning: Could not parse value '{val}' for schema 'binary_good_bad' - Error: {e}")
            return None


    elif schema_name == "ternary_bad_medium_good":
        # Model: Bad=0, Medium=1, Good=2 ; GT: 1-2=0, 3=1, 4-5=2
        if val.lower() == "bad":
            return 0
        elif val.lower() == "medium":
            return 1
        elif val.lower() == "good":
            return 2
        else:
            try:
                iv = int(val)
                if iv in [1, 2]:
                    return 0
                elif iv == 3:
                    return 1
                elif iv in [4, 5]:
                    return 2
            except:
                pass
        return None

    elif schema_name == "binary_effective_ineffective":
        # Model: Effective=0, Ineffective=1 ; GT similar (assumed)
        if val.lower() == "effective":
            return 0
        elif val.lower() == "ineffective":
            return 1
        else:
            # try to parse as integer
            try:
                iv = int(val)
                if iv in [1, 2]:
                    return 0
                elif iv in [3,4,5]:
                    return 1
            except:
                pass
        return None

    else:
        # If the schema is not recognized, return None
        return None

# prepares scores for a specific dimension from the data
def prepare_scores(data, dim, schema_name):
    scores = []
    for x in data:
        val = normalize_for_dimension(x[dim], schema_name, dim)
        if val is not None:
            scores.append(val)
        else:
            print(f"Warning: '{x[dim]}' is not in expected classes for schema '{schema_name}' and dimension '{dim}'")
            scores.append(-1)
    return scores

# prints a confusion matrix in a dynamic format
def print_dynamic_cm(cm, labels):
    print("\nConfusion Matrix (Actual vs Predicted):")
    if cm.shape[0] != len(labels):
        print(f"Warning: mismatch between matrix size {cm.shape} and labels length {len(labels)}")
        return
    header = " ".join(f"{l:>8}" for l in labels)
    print(f"{'':10}{header}")
    for i, row in enumerate(cm):
        row_str = " ".join(f"{val:8.2f}" for val in row)
        print(f"{labels[i]:10}{row_str}")

# Evaluates a single run against the ground truth
def evaluate_single_run(model_outputs, ground_truths, schema_name="binary_good_bad"):
    print("\n --- EVALUATION RESULTS (SINGLE RUN) ---\n")
    for dim in dimensions:
        true_scores = prepare_scores(ground_truths, dim, schema_name)
        model_scores = prepare_scores(model_outputs, dim, schema_name)

        paired = [(t, p) for t, p in zip(true_scores, model_scores) if t >= 0 and p >= 0]
        if not paired:
            print(f"No valid samples for {dim}, skipping.")
            continue

        y_true = [t for t, p in paired]
        y_pred = [p for t, p in paired]

        classes = sorted(set(y_true + y_pred))
        cm = confusion_matrix(y_true, y_pred, labels=classes)
        report = classification_report(y_true, y_pred, labels=classes, zero_division=0)

        print(f"\n --- {dim.upper()} ---")
        print_dynamic_cm(cm, [str(c) for c in classes])
        print("\nClassification Report:")
        print(report)

# Confusion matrix and standard deviation across multiple runs
def compute_avg_cm_and_std(model_outputs_runs, ground_truths, schema_name):
    n_runs = len(model_outputs_runs)
    avg_cms = {}

    for dim in dimensions:
        all_classes = set()

        # First: prepare all classes from the ground truth and model outputs
        true_scores_total = prepare_scores(ground_truths, dim, schema_name)
        for run_outputs in model_outputs_runs:
            model_scores_total = prepare_scores(run_outputs, dim, schema_name)
            paired_total = [(t, p) for t, p in zip(true_scores_total, model_scores_total) if t >= 0 and p >= 0]
            for t, p in paired_total:
                all_classes.update([t, p])

        all_classes = sorted(all_classes)
        if not all_classes:
            avg_cms[dim] = {"mean_cm": None, "std_cm": None, "labels": []}
            continue

        cm_stack = []

        # Second: calculate confusion matrices for each run
        for run_outputs in model_outputs_runs:
            true_scores = prepare_scores(ground_truths, dim, schema_name)
            model_scores = prepare_scores(run_outputs, dim, schema_name)
            paired = [(t, p) for t, p in zip(true_scores, model_scores) if t >= 0 and p >= 0]
            if not paired:
                continue
            y_true = [t for t, p in paired]
            y_pred = [p for t, p in paired]
            cm = confusion_matrix(y_true, y_pred, labels=all_classes)
            cm_stack.append(cm)

        if cm_stack:
            cm_array = np.stack(cm_stack, axis=2)  # Shape: (n_classes, n_classes, n_runs)
            mean_cm = np.mean(cm_array, axis=2)
            std_cm = np.std(cm_array, axis=2)
            avg_cms[dim] = {"mean_cm": mean_cm, "std_cm": std_cm, "labels": [str(c) for c in all_classes]}
        else:
            avg_cms[dim] = {"mean_cm": None, "std_cm": None, "labels": [str(c) for c in all_classes]}

    return avg_cms


# mean report across multiple runs
def compute_avg_report(model_outputs_runs, ground_truths, schema_name):
    n_runs = len(model_outputs_runs)
    avg_reports = {}

    for dim in dimensions:
        total_report = defaultdict(lambda: defaultdict(float))
        classes = None

        for run_outputs in model_outputs_runs:
            true_scores = prepare_scores(ground_truths, dim, schema_name)
            model_scores = prepare_scores(run_outputs, dim, schema_name)
            paired = [(t, p) for t, p in zip(true_scores, model_scores) if t >= 0 and p >= 0]
            if not paired:
                continue
            y_true = [t for t, p in paired]
            y_pred = [p for t, p in paired]
            run_classes = sorted(set(y_true + y_pred))
            if classes is None:
                classes = run_classes
            report = classification_report(y_true, y_pred, labels=run_classes, output_dict=True, zero_division=0)
            for label, metrics in report.items():
                if isinstance(metrics, dict):
                    for metric_name, value in metrics.items():
                        total_report[label][metric_name] += value

        averaged = {}
        for label, metrics in total_report.items():
            averaged[label] = {k: v / n_runs for k, v in metrics.items()}
        avg_reports[dim] = (averaged, [str(c) for c in classes] if classes else [])

    return avg_reports

# Evaluate multiple runs against the ground truth
def evaluate_multiple_runs(model_outputs_runs, ground_truths, schema_name):
    print("\n === AGGREGATED ANALYSIS OVER MULTIPLE RUNS ===\n")
    
    avg_cms = compute_avg_cm_and_std(model_outputs_runs, ground_truths, schema_name)
    for dim in dimensions:
        print(f"\n --- {dim.upper()} ---")
        cm_data = avg_cms[dim]

        if cm_data["mean_cm"] is not None:
            print("\nAverage Confusion Matrix (mean across runs):")
            print_dynamic_cm(cm_data["mean_cm"], cm_data["labels"])

            print("\nStandard Deviation of Confusion Matrix across runs:")
            print_dynamic_cm(cm_data["std_cm"], cm_data["labels"])
        else:
            print("Not enough data for this dimension.")

    print("\n === AVERAGE CLASSIFICATION REPORT ===")
    avg_reports = compute_avg_report(model_outputs_runs, ground_truths, schema_name)
    for dim in dimensions:
        print(f"\n --- {dim.upper()} ---")
        report, classes = avg_reports[dim]
        print(f"{'Label':<20}{'precision':>10}{'recall':>10}{'f1-score':>10}{'support':>10}")
        print("-" * 60)
        for label in report:
            m = report[label]
            if isinstance(m, dict):
                print(f"{label:<20}{m.get('precision', 0):10.2f}{m.get('recall', 0):10.2f}{m.get('f1-score', 0):10.2f}{m.get('support', 0):10.0f}")

# Analyze variability across runs
def analyze_variability_across_runs(runs_outputs, ground_truths, schema_name):
    print("\n --- VARIABILITY RESULTS BETWEEN RUNS ---\n")

    n_args = len(runs_outputs[0])
    n_runs = len(runs_outputs)

    for dim in dimensions:
        print(f"\n --- VARIABILITY IN {dim.upper()} ---")
        gt_scores = prepare_scores(ground_truths, dim, schema_name)

        bin_matrix = np.array([
            [
                normalize_for_dimension(run[i][dim], schema_name, dim)
                for run in runs_outputs
            ]
            for i in range(n_args)
        ])

        std_by_argument = np.std(bin_matrix, axis=1)
        mean_std = mean(std_by_argument)

        disagreement_flags = [len(set(row)) > 1 for row in bin_matrix]
        num_disagreements = sum(disagreement_flags)
        proportion_disagreement = num_disagreements / n_args

        print(f"\nVariability between runs:")
        print(f"Mean standard deviation per argument: {mean_std:.2f}")
        print(f"Arguments with disagreement: {num_disagreements} / {n_args} ({proportion_disagreement:.2%})")

        matches_per_argument = []
        for i in range(n_args):
            gt = gt_scores[i]
            if gt < 0:
                continue
            run_preds = bin_matrix[i]
            num_matches = sum(1 for pred in run_preds if pred == gt)
            matches_per_argument.append(num_matches)

        total_possible = n_runs * n_args
        total_matches = sum(matches_per_argument)
        overall_accuracy = total_matches / total_possible if total_possible > 0 else 0

        mean_accuracy_per_argument = mean([m / n_runs for m in matches_per_argument])
        std_accuracy_per_argument = np.std([m / n_runs for m in matches_per_argument])

        fully_correct = sum(1 for m in matches_per_argument if m == n_runs)
        medium_correct = sum(1 for m in matches_per_argument if n_runs * 0.5 <= m < n_runs)
        low_correct = sum(1 for m in matches_per_argument if m < n_runs * 0.5)

        print(f"\nOverall accuracy vs Ground Truth (mean across all runs and arguments): {overall_accuracy:.2%}")
        print(f"Mean accuracy per argument: {mean_accuracy_per_argument:.2%}")
        print(f"Standard deviation of accuracy per argument: {std_accuracy_per_argument:.2%}")

        print(f"\nDistribution of arguments by run accuracy rate:")
        print(f"- {fully_correct} arguments ({fully_correct / n_args:.2%}) fully correct in 100% of runs.")
        print(f"- {medium_correct} arguments ({medium_correct / n_args:.2%}) correct in 50% to 99% of runs.")
        print(f"- {low_correct} arguments ({low_correct / n_args:.2%}) correct in less than 50% of runs.")

        print("\nPrediction matrix by argument (rows=arguments, columns=runs):")
        print(bin_matrix)

        rounded_means = np.rint(np.mean(bin_matrix, axis=1)).astype(int)
        valid_indices = [i for i, gt in enumerate(gt_scores) if gt >= 0]

        correct_avg_preds = sum(1 for i in valid_indices if rounded_means[i] == gt_scores[i])
        accuracy_avg_preds = correct_avg_preds / len(valid_indices) if valid_indices else 0

        print(f"\nAccuracy using rounded average of predictions per argument vs Ground Truth: {accuracy_avg_preds:.2%}")

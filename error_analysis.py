import json
import os
import numpy as np
import matplotlib.pyplot as plt
import csv
from dataset_division import test_data

dimensions = ["cogency", "effectiveness", "reasonableness", "overall"]

def get_threshold(dim):
    return 3 if dim == "reasonableness" else 3.33

def normalize_error(error, dim, error_type):
    if dim == "reasonableness":
        if error_type == "bad_to_good":
            min_err, max_err = 2, 4
            return (error - min_err) / (max_err - min_err)
        else:  # good_to_bad
            min_err, max_err = -4, -2
            return (max_err - error) / (max_err - min_err)
    else:
        if error_type == "bad_to_good":
            min_err, max_err = 2.33, 4
            return (error - min_err) / (max_err - min_err)
        else:  # good_to_bad
            min_err, max_err = -4, -1.67
            return (max_err - error) / (max_err - min_err)

def analyze_error_severity(model_runs, ground_truths, output_folder):
    all_errors_by_dim = {}
    error_records_by_dim = {}

    all_bad_to_good = {}
    all_good_to_bad = {}

    for dim in dimensions:
        print(f"\n=== Error Severity for {dim.upper()} ===")
        threshold = get_threshold(dim)
        all_errors = []
        error_records = []

        errors_bad_to_good = []
        errors_good_to_bad = []

        for run_idx, run in enumerate(model_runs):
            for i, pred_item in enumerate(run):
                if pred_item is None:
                    continue
                gt_val = ground_truths[i][dim]
                pred_label = pred_item[dim]

                try:
                    gt_float = float(gt_val)
                except:
                    continue

                gt_bin = 1 if gt_float >= threshold else 0
                pred_bin = 1 if pred_label.lower() == "good" else 0

                if gt_bin != pred_bin:
                    error_type = "bad_to_good" if (pred_bin == 0 and gt_bin == 1) else "good_to_bad"
                    target_numeric = 1 if error_type == "bad_to_good" else 5
                    error = gt_float - target_numeric
                    norm_error = normalize_error(error, dim, error_type)
                    norm_error = max(0, min(1, norm_error))

                    all_errors.append(norm_error)
                    error_records.append({
                        "Run": run_idx + 1,
                        "Argument_Index": i,
                        "Ground_Truth_Score": gt_float,
                        "Ground_Truth_Binary": gt_bin,
                        "Predicted_Label": pred_label,
                        "Predicted_Binary": pred_bin,
                        "Error": error,
                        "Normalized_Error": norm_error,
                        "Error_Type": error_type
                    })

                    if error_type == "bad_to_good":
                        errors_bad_to_good.append(norm_error)
                    else:
                        errors_good_to_bad.append(norm_error)

        all_errors_by_dim[dim] = all_errors
        error_records_by_dim[dim] = error_records

        # Guardamos para gráfico final
        all_bad_to_good[dim] = errors_bad_to_good
        all_good_to_bad[dim] = errors_good_to_bad

        csv_filename = os.path.join(output_folder, f"{dim}_error_analysis.csv")
        with open(csv_filename, mode='w', newline='') as csv_file:
            fieldnames = ["Run", "Argument_Index", "Ground_Truth_Score", "Ground_Truth_Binary",
                          "Predicted_Label", "Predicted_Binary", "Error", "Normalized_Error", "Error_Type"]
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()
            for record in error_records:
                writer.writerow(record)

        print(f"Total errors for {dim}: {len(all_errors)}")
        if all_errors:
            print(f"Mean normalized error: {np.mean(all_errors):.3f}")
            print(f" Std deviation: {np.std(all_errors):.3f}")
        else:
            print(f"No errors found for {dim} dimension.")

        # Gráficos individuales por dimensión
        # fig, axs = plt.subplots(1, 2, figsize=(10, 4))
        # axs[0].hist(errors_bad_to_good, bins=10, range=(0, 1), alpha=0.7, color='coral', edgecolor='black')
        # axs[0].set_title(f"{dim.upper()} - Pred Good, GT Bad")
        # axs[0].set_xlabel("Normalized Error Severity")
        # axs[0].set_ylabel("Frequency")

        # axs[1].hist(errors_good_to_bad, bins=10, range=(0, 1), alpha=0.7, color='skyblue', edgecolor='black')
        # axs[1].set_title(f"{dim.upper()} - Pred Bad, GT Good")
        # axs[1].set_xlabel("Normalized Error Severity")
        # axs[1].set_ylabel("Frequency")

        # plt.tight_layout()
        # plot_filename = os.path.join(output_folder, f"{dim}_error_severity_separated.png")
        # plt.savefig(plot_filename)
        # plt.show()

    # Gráfico compuesto con 8 subplots (4 filas x 2 columnas)
    fig, axs = plt.subplots(len(dimensions), 2, figsize=(12, 16))

    for i, dim in enumerate(dimensions):
        axs[i, 0].hist(all_bad_to_good[dim], bins=10, range=(0, 1), alpha=0.7, color='coral', edgecolor='black')
        axs[i, 0].set_title(f"{dim.upper()} - Pred Good, GT Bad")
        axs[i, 0].set_xlabel("Normalized Error Severity")
        axs[i, 0].set_ylabel("Frequency")

        axs[i, 1].hist(all_good_to_bad[dim], bins=10, range=(0, 1), alpha=0.7, color='skyblue', edgecolor='black')
        axs[i, 1].set_title(f"{dim.upper()} - Pred Bad, GT Good")
        axs[i, 1].set_xlabel("Normalized Error Severity")
        axs[i, 1].set_ylabel("Frequency")

    plt.tight_layout()
    plot_filename_all = os.path.join(output_folder, "error_severity_8plots.png")
    plt.savefig(plot_filename_all)
    plt.show()


# --------- MAIN ----------
response_dir = "model_responses"
response_files = [f for f in os.listdir(response_dir) if f.startswith("model_responses_") and f.endswith(".json")]
if not response_files:
    print("No model response files found.")
    exit()

print("\nAvailable model responses files:")
for i, f in enumerate(response_files):
    print(f"{i + 1}: {f}")

selected_idx = int(input("Select a file by number: ")) - 1
if selected_idx < 0 or selected_idx >= len(response_files):
    print("Invalid selection.")
    exit()

selected_filename = response_files[selected_idx]
with open(selected_filename, 'r') as f:
    all_runs = json.load(f)

ground_truths = [entry["labels"] for entry in test_data]
print(f"\nLoaded {len(all_runs)} runs with {len(all_runs[0])} predictions each.")

# Generate subfolder name
base_name = os.path.splitext(selected_filename)[0].replace("model_responses_", "")
output_folder = os.path.join("error_analysis_plots", f"error_{base_name}")
os.makedirs(output_folder, exist_ok=True)

analyze_error_severity(all_runs, ground_truths, output_folder)

print(f"\n--- Error severity analysis finished ---")
print(f"CSV files and plot saved in folder: {output_folder}")

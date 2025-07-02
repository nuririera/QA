import json
import os
import numpy as np
import matplotlib.pyplot as plt
import csv
from dataset_division import test_data

dimensions = ["cogency", "effectiveness", "reasonableness", "overall"]

def get_threshold(dim):
    return 3 if dim =="reasonableness" else 3.33

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
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for dim in dimensions:
        print(f"\n=== Error Severity for {dim.upper()} ===")
        threshold = get_threshold(dim)
        all_errors = []
        error_records = []

        for run_idx, run in enumerate(model_runs):
            for i, pred_item in enumerate(run):
                gt_val = ground_truths[i][dim]
                pred_label = pred_item[dim]

                try:
                    gt_float = float(gt_val)
                except:
                    continue
            
                # ground truth binarization
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

            # Plotting the distribution of normalized errors
            plt.figure(figsize=(6, 4))
            plt.hist(all_errors, bins=10, range=(0, 1), alpha=0.7, color='coral', edgecolor='black')
            plt.title(f"Error severity distribution for {dim.upper()}")
            plt.ylabel("Frequency")
            plt.xlabel("Normalized Error severity (0 = min error, 1 = max error)")
            plt.tight_layout()
            plot_filename = os.path.join(output_folder, f"error_severity_{dim}.png")
            plt.savefig(plot_filename)
            plt.show()

        else:
            print(f"No errors found for {dim} dimension.")

# --------- MAIN ----------
response_files = [f for f in os.listdir() if f.startswith("model_responses_") and f.endswith(".json")]
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

with open(response_files[selected_idx], 'r') as f:
    all_runs = json.load(f)

ground_truths = [entry["labels"] for entry in test_data]
print(f"\nLoaded {len(all_runs)} runs with {len(all_runs[0])} predictions each.")

output_folder = "error_analysis_plots"

analyze_error_severity(all_runs, ground_truths, output_folder)

print(f"\n--- Error severity analysis finished ---")
print(f"CSV files and plots saved in folder: {output_folder}")

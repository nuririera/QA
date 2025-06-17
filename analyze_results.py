import numpy as np
from statistics import mean, stdev
from sklearn.metrics import f1_score, confusion_matrix, classification_report

# Define the mapping for ratings
# The ratings are "Good" and "Bad", we map them to 1 and 0 respectively
rating_map = {"Good": 1, "Bad": 0}

# Function to analyze the results of the model outputs against ground truths
def evaluate_single_run(model_outpus, ground_truths):
    print("\n --- ANALYSIS RESULTS --- (evaluating model outputs against ground truths - 1 run)\n")
    # List of dimensions to analyze
    dimensions = ["cogency", "effectiveness", "reasonableness", "overall"]

    for dim in dimensions:
        # Turn the dimension into a key for accessing the ratings
        model_scores = [rating_map[x[dim]] for x in model_outpus]
        true_scores = [rating_map[x[dim]] for x in ground_truths]

        print(f"\n --- {dim.upper()} ---")
        # Metric of basic classification performance
        print(f"F1 Score: {f1_score(true_scores, model_scores):.2f}")
        print(f"Matriz de confusión:\n{confusion_matrix(true_scores, model_scores)}")
        print("Classification Report:")
        print(classification_report(true_scores, model_scores, zero_division=0))

def analyze_variability_across_runs(runs_outputs):
    print("\n --- ANALYSIS RESULTS --- (analyzing variability across multiple runs)\n")
    dimensions = ["cogency", "effectiveness", "reasonableness", "overall"]

    n_runs = len(runs_outputs)
    n_args = len(runs_outputs[0])

    for dim in dimensions:
        print(f"\n --- VARIABILIDAD EN {dim.upper()} ---")
        # bin_matrix[i][j] = evaluación binaria del argumento i en el run j
        bin_matrix = np.array([
            [rating_map[run[i][dim]] for run in runs_outputs]
            for i in range(n_args)
        ])

        # Varianza por argumento: mide cuánto varían los runs para ese argumento
        var_by_argument = np.var(bin_matrix, axis=1)
        mean_var = mean(var_by_argument)

        print(f"Varianza media por argumento: {mean_var:.2f}")

        # Proporción de argumentos con desacuerdo entre runs
        desacuerdo = 1 - np.mean([
            len(set(row)) == 1 for row in bin_matrix
        ])

        print(f"Proporción de argumentos con desacuerdo entre runs: {desacuerdo:.2%}")

        print("Ejemplo primeros 5 argumentos (filas = argumentos, columnas = runs):")
        print(bin_matrix)

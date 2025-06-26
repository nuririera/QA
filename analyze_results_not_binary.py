import numpy as np
from statistics import mean
from sklearn.metrics import confusion_matrix, classification_report
from collections import defaultdict

dimensions = ["cogency", "effectiveness", "reasonableness", "overall"]

# Normaliza según el esquema seleccionado
def normalize_for_dimension(value, schema_name):
    val = str(value).strip()

    if schema_name == "numeric_1_to_5":
        # Valores del 1 al 5 numéricos
        try:
            iv = int(val)
            if 1 <= iv <= 5:
                return iv
            else:
                return None
        except:
            return None

    elif schema_name == "binary_good_bad":
    # Modelo: Good=0, Bad=1 ; GT: Bad si media < 3.5, Good si >= 3.5

        if val is None:
            return None

        try:
            # Si es texto "Good"/"Bad"
            if isinstance(val, str):
                if val.lower() == "good":
                    return 0
                elif val.lower() == "bad":
                    return 1
                else:
                    # Intentar convertir strings numéricos como '3.3', '4.6666'...
                    fv = float(val)
            else:
                # Si ya es numérico (int o float)
                fv = float(val)

            # Aplicar el umbral
            if fv < 3.3:
                return 1  # Bad
            else:
                return 0  # Good

        except Exception as e:
            print(f"Warning: Could not parse value '{val}' for schema 'binary_good_bad' - Error: {e}")
            return None


    elif schema_name == "ternary_bad_medium_good":
        # Modelo: Bad=0, Medium=1, Good=2 ; GT: 1-2=0, 3=1, 4-5=2
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
        # Modelo: Effective=0, Ineffective=1 ; GT similar (asumimos)
        if val.lower() == "effective":
            return 0
        elif val.lower() == "ineffective":
            return 1
        else:
            # Intentamos numérico para GT
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
        # Si esquema no reconocido, devuelve None
        return None

# Prepara lista de puntuaciones normalizadas
def prepare_scores(data, dim, schema_name):
    scores = []
    for x in data:
        val = normalize_for_dimension(x[dim], schema_name)
        if val is not None:
            scores.append(val)
        else:
            print(f"Warning: '{x[dim]}' no está en clases esperadas para esquema '{schema_name}' y dimensión '{dim}'")
            scores.append(-1)
    return scores

# Imprime matriz de confusión dinámica
def print_dynamic_cm(cm, labels):
    print("\nConfusion Matrix (Actual vs Predicted):")
    if cm.shape[0] != len(labels):
        print(f"Warning: mismatch entre tamaño matriz {cm.shape} y longitud labels {len(labels)}")
        return
    header = " ".join(f"{l:>8}" for l in labels)
    print(f"{'':10}{header}")
    for i, row in enumerate(cm):
        row_str = " ".join(f"{val:8.2f}" for val in row)
        print(f"{labels[i]:10}{row_str}")

# Evalúa una ejecución simple
def evaluate_single_run(model_outputs, ground_truths, schema_name):
    print("\n --- RESULTADOS DE EVALUACIÓN (SINGLE RUN) ---\n")
    for dim in dimensions:
        true_scores = prepare_scores(ground_truths, dim, schema_name)
        model_scores = prepare_scores(model_outputs, dim, schema_name)

        paired = [(t, p) for t, p in zip(true_scores, model_scores) if t >= 0 and p >= 0]
        if not paired:
            print(f"No hay muestras válidas para {dim}, saltando.")
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

# Matriz de confusión promedio en múltiples ejecuciones
def compute_avg_cm_and_std(model_outputs_runs, ground_truths, schema_name):
    n_runs = len(model_outputs_runs)
    avg_cms = {}

    for dim in dimensions:
        all_classes = set()

        # Primero: Reunir todas las clases posibles a lo largo de todos los runs y el ground truth
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

        # Segundo: Calcular la confusion matrix por run usando las mismas clases
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


# Reporte promedio en múltiples ejecuciones
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

# Evalúa múltiples ejecuciones
def evaluate_multiple_runs(model_outputs_runs, ground_truths, schema_name):
    print("\n === ANÁLISIS AGREGADO EN MÚLTIPLES EJECUCIONES ===\n")
    
    avg_cms = compute_avg_cm_and_std(model_outputs_runs, ground_truths, schema_name)
    for dim in dimensions:
        print(f"\n --- {dim.upper()} ---")
        cm_data = avg_cms[dim]

        if cm_data["mean_cm"] is not None:
            print("\nMatriz de Confusión Promedio (media entre runs):")
            print_dynamic_cm(cm_data["mean_cm"], cm_data["labels"])

            print("\nDesviación Estándar de la Matriz de Confusión entre runs:")
            print_dynamic_cm(cm_data["std_cm"], cm_data["labels"])
        else:
            print("No hay datos suficientes para esta dimensión.")

    print("\n === REPORTE PROMEDIO DE CLASIFICACIÓN ===")
    avg_reports = compute_avg_report(model_outputs_runs, ground_truths, schema_name)
    for dim in dimensions:
        print(f"\n --- {dim.upper()} ---")
        report, classes = avg_reports[dim]
        print(f"{'Etiqueta':<20}{'precision':>10}{'recall':>10}{'f1-score':>10}{'support':>10}")
        print("-" * 60)
        for label in report:
            m = report[label]
            if isinstance(m, dict):
                print(f"{label:<20}{m.get('precision', 0):10.2f}{m.get('recall', 0):10.2f}{m.get('f1-score', 0):10.2f}{m.get('support', 0):10.0f}")

# Analiza variabilidad entre ejecuciones
def analyze_variability_across_runs(runs_outputs, ground_truths, schema_name):
    print("\n --- RESULTADOS DE VARIABILIDAD ENTRE EJECUCIONES ---\n")

    n_args = len(runs_outputs[0])
    n_runs = len(runs_outputs)

    for dim in dimensions:
        print(f"\n --- VARIABILIDAD EN {dim.upper()} ---")
        gt_scores = prepare_scores(ground_truths, dim, schema_name)

        bin_matrix = np.array([
            [
                normalize_for_dimension(run[i][dim], schema_name)
                for run in runs_outputs
            ]
            for i in range(n_args)
        ])

        std_by_argument = np.std(bin_matrix, axis=1)
        mean_std = mean(std_by_argument)

        disagreement_flags = [len(set(row)) > 1 for row in bin_matrix]
        num_disagreements = sum(disagreement_flags)
        proportion_disagreement = num_disagreements / n_args

        print(f"\nVariabilidad entre runs:")
        print(f"Media de desviación estándar por argumento: {mean_std:.2f}")
        print(f"Argumentos con desacuerdo: {num_disagreements} / {n_args} ({proportion_disagreement:.2%})")

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

        print(f"\nPrecisión global respecto al Ground Truth (media sobre todos los runs y argumentos): {overall_accuracy:.2%}")
        print(f"Precisión media por argumento: {mean_accuracy_per_argument:.2%}")
        print(f"Desviación estándar de precisión por argumento: {std_accuracy_per_argument:.2%}")

        print(f"\nDistribución de argumentos según tasa de aciertos en runs:")
        print(f"- {fully_correct} argumentos ({fully_correct / n_args:.2%}) acertados en 100% de los runs.")
        print(f"- {medium_correct} argumentos ({medium_correct / n_args:.2%}) con entre 50% y 99% de aciertos.")
        print(f"- {low_correct} argumentos ({low_correct / n_args:.2%}) con menos del 50% de aciertos.")

        print("\nMatriz de predicciones por argumento (filas=argumentos, columnas=runs):")
        print(bin_matrix)

        rounded_means = np.rint(np.mean(bin_matrix, axis=1)).astype(int)
        valid_indices = [i for i, gt in enumerate(gt_scores) if gt >= 0]

        correct_avg_preds = sum(1 for i in valid_indices if rounded_means[i] == gt_scores[i])
        accuracy_avg_preds = correct_avg_preds / len(valid_indices) if valid_indices else 0

        print(f"\nPrecisión usando la media redondeada de predicciones por argumento vs Ground Truth: {accuracy_avg_preds:.2%}")

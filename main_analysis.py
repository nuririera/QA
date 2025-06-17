from analyze_results import evaluate_single_run, analyze_variability_across_runs

# Solo para test: 2 runs de 3 argumentos
run1 = [
    {"cogency": "Good", "effectiveness": "Good", "reasonableness": "Bad", "overall": "Good"},
    {"cogency": "Bad", "effectiveness": "Bad", "reasonableness": "Good", "overall": "Bad"},
    {"cogency": "Good", "effectiveness": "Bad", "reasonableness": "Bad", "overall": "Bad"},
    {"cogency": "Bad", "effectiveness": "Bad", "reasonableness": "Good", "overall": "Bad"},
    {"cogency": "Good", "effectiveness": "Bad", "reasonableness": "Bad", "overall": "Bad"},
    {"cogency": "Bad", "effectiveness": "Bad", "reasonableness": "Good", "overall": "Bad"},
    {"cogency": "Good", "effectiveness": "Bad", "reasonableness": "Bad", "overall": "Bad"},
    {"cogency": "Bad", "effectiveness": "Bad", "reasonableness": "Good", "overall": "Bad"},
    {"cogency": "Good", "effectiveness": "Bad", "reasonableness": "Bad", "overall": "Bad"},
    {"cogency": "Bad", "effectiveness": "Bad", "reasonableness": "Good", "overall": "Bad"},
    {"cogency": "Good", "effectiveness": "Bad", "reasonableness": "Bad", "overall": "Bad"}
]
run2 = [
    {"cogency": "Good", "effectiveness": "Good", "reasonableness": "Bad", "overall": "Good"},
    {"cogency": "Bad", "effectiveness": "Good", "reasonableness": "Good", "overall": "Bad"},
    {"cogency": "Bad", "effectiveness": "Bad", "reasonableness": "Good", "overall": "Good"},
     {"cogency": "Bad", "effectiveness": "Bad", "reasonableness": "Good", "overall": "Bad"},
    {"cogency": "Good", "effectiveness": "Bad", "reasonableness": "Bad", "overall": "Bad"},
    {"cogency": "Bad", "effectiveness": "Bad", "reasonableness": "Good", "overall": "Bad"},
    {"cogency": "Good", "effectiveness": "Bad", "reasonableness": "Bad", "overall": "Bad"},
    {"cogency": "Bad", "effectiveness": "Bad", "reasonableness": "Good", "overall": "Bad"},
    {"cogency": "Good", "effectiveness": "Bad", "reasonableness": "Bad", "overall": "Bad"},
    {"cogency": "Bad", "effectiveness": "Bad", "reasonableness": "Good", "overall": "Bad"},
    {"cogency": "Good", "effectiveness": "Bad", "reasonableness": "Bad", "overall": "Bad"}
]

run3 = [
    {"cogency": "Good", "effectiveness": "Good", "reasonableness": "Bad", "overall": "Good"},
    {"cogency": "Bad", "effectiveness": "Good", "reasonableness": "Good", "overall": "Bad"},
    {"cogency": "Bad", "effectiveness": "Bad", "reasonableness": "Good", "overall": "Good"},
    {"cogency": "Bad", "effectiveness": "Bad", "reasonableness": "Good", "overall": "Bad"},
    {"cogency": "Good", "effectiveness": "Bad", "reasonableness": "Bad", "overall": "Bad"},
    {"cogency": "Bad", "effectiveness": "Bad", "reasonableness": "Good", "overall": "Bad"},
    {"cogency": "Good", "effectiveness": "Bad", "reasonableness": "Bad", "overall": "Bad"},
    {"cogency": "Bad", "effectiveness": "Bad", "reasonableness": "Good", "overall": "Bad"},
    {"cogency": "Good", "effectiveness": "Bad", "reasonableness": "Bad", "overall": "Bad"},
    {"cogency": "Bad", "effectiveness": "Bad", "reasonableness": "Good", "overall": "Bad"},
    {"cogency": "Good", "effectiveness": "Bad", "reasonableness": "Bad", "overall": "Bad"}
]

run4 = [
    {"cogency": "Good", "effectiveness": "Good", "reasonableness": "Bad", "overall": "Good"},
    {"cogency": "Bad", "effectiveness": "Bad", "reasonableness": "Good", "overall": "Bad"},
    {"cogency": "Good", "effectiveness": "Bad", "reasonableness": "Bad", "overall": "Bad"},
    {"cogency": "Bad", "effectiveness": "Bad", "reasonableness": "Good", "overall": "Bad"},
    {"cogency": "Good", "effectiveness": "Bad", "reasonableness": "Bad", "overall": "Bad"},
    {"cogency": "Bad", "effectiveness": "Bad", "reasonableness": "Good", "overall": "Bad"},
    {"cogency": "Good", "effectiveness": "Bad", "reasonableness": "Bad", "overall": "Bad"},
    {"cogency": "Bad", "effectiveness": "Bad", "reasonableness": "Good", "overall": "Bad"},
    {"cogency": "Good", "effectiveness": "Bad", "reasonableness": "Bad", "overall": "Bad"},
    {"cogency": "Bad", "effectiveness": "Bad", "reasonableness": "Good", "overall": "Bad"},
    {"cogency": "Good", "effectiveness": "Bad", "reasonableness": "Bad", "overall": "Bad"}
]

run5 = [
    {"cogency": "Good", "effectiveness": "Good", "reasonableness": "Bad", "overall": "Good"},
    {"cogency": "Bad", "effectiveness": "Bad", "reasonableness": "Good", "overall": "Bad"},
    {"cogency": "Good", "effectiveness": "Bad", "reasonableness": "Bad", "overall": "Bad"},
     {"cogency": "Bad", "effectiveness": "Bad", "reasonableness": "Good", "overall": "Bad"},
    {"cogency": "Good", "effectiveness": "Good", "reasonableness": "Bad", "overall": "Bad"},
    {"cogency": "Bad", "effectiveness": "Bad", "reasonableness": "Good", "overall": "Bad"},
    {"cogency": "Good", "effectiveness": "Bad", "reasonableness": "Bad", "overall": "Bad"},
    {"cogency": "Bad", "effectiveness": "Bad", "reasonableness": "Bad", "overall": "Bad"},
    {"cogency": "Good", "effectiveness": "Bad", "reasonableness": "Bad", "overall": "Bad"},
    {"cogency": "Bad", "effectiveness": "Bad", "reasonableness": "Good", "overall": "Bad"},
    {"cogency": "Good", "effectiveness": "Bad", "reasonableness": "Bad", "overall": "Bad"}
]

run6 = [
    {"cogency": "Good", "effectiveness": "Good", "reasonableness": "Bad", "overall": "Good"},
    {"cogency": "Bad", "effectiveness": "Bad", "reasonableness": "Good", "overall": "Bad"},
    {"cogency": "Good", "effectiveness": "Good", "reasonableness": "Bad", "overall": "Bad"},
     {"cogency": "Bad", "effectiveness": "Bad", "reasonableness": "Good", "overall": "Bad"},
    {"cogency": "Good", "effectiveness": "Good", "reasonableness": "Bad", "overall": "Bad"},
    {"cogency": "Bad", "effectiveness": "Bad", "reasonableness": "Bad", "overall": "Bad"},
    {"cogency": "Good", "effectiveness": "Bad", "reasonableness": "Bad", "overall": "Bad"},
    {"cogency": "Bad", "effectiveness": "Bad", "reasonableness": "Good", "overall": "Good"},
    {"cogency": "Bad", "effectiveness": "Bad", "reasonableness": "Good", "overall": "Bad"},
    {"cogency": "Bad", "effectiveness": "Bad", "reasonableness": "Good", "overall": "Bad"},
    {"cogency": "Bad", "effectiveness": "Bad", "reasonableness": "Bad", "overall": "Bad"}
]

ground_truth = [
    {"cogency": "Good", "effectiveness": "Good", "reasonableness": "Bad", "overall": "Good"},
    {"cogency": "Bad", "effectiveness": "Bad", "reasonableness": "Good", "overall": "Bad"},
    {"cogency": "Good", "effectiveness": "Good", "reasonableness": "Good", "overall": "Good"},
     {"cogency": "Bad", "effectiveness": "Bad", "reasonableness": "Bad", "overall": "Bad"},
    {"cogency": "Good", "effectiveness": "Bad", "reasonableness": "Bad", "overall": "Bad"},
    {"cogency": "Good", "effectiveness": "Bad", "reasonableness": "Good", "overall": "Bad"},
    {"cogency": "Good", "effectiveness": "Bad", "reasonableness": "Bad", "overall": "Bad"},
    {"cogency": "Bad", "effectiveness": "Good", "reasonableness": "Good", "overall": "Bad"},
    {"cogency": "Good", "effectiveness": "Bad", "reasonableness": "Bad", "overall": "Bad"},
    {"cogency": "Bad", "effectiveness": "Bad", "reasonableness": "Bad", "overall": "Bad"},
    {"cogency": "Good", "effectiveness": "Bad", "reasonableness": "Bad", "overall": "Bad"}
]

# Run evaluaciones
evaluate_single_run(run1, ground_truth)
evaluate_single_run(run2, ground_truth)
evaluate_single_run(run3, ground_truth)
evaluate_single_run(run4, ground_truth)
evaluate_single_run(run5, ground_truth)
evaluate_single_run(run6, ground_truth)
analyze_variability_across_runs([run1, run2, run3, run4, run5, run6])

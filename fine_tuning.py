import pandas as pd
import json

# Reglas de mapeo (con umbral distinto para Reasonableness)
def map_scores(dimension, score):
    if dimension == 'reasonableness':
        return 'Good' if score >= 3.0 else 'Bad'
    else:
        return 'Good' if score >= 3.33 else 'Bad'

# Construcción del prompt completo por argumento
def build_prompt(argument_text, output_json):
    text = f"""###ROLE### You are an Argument Annotator AI. ###OBJECTIVE### Your task is to asses the quality of an argument across four dimensions: cogency, effectiveness, reasonableness and overall. For each dimension, provide a binary score: - "Bad" - "Good" ###INSTRUCTIONS### You must evaluate each dimension independently, based strictly on the provided definitions. Be particularly strict and conservative when evaluating. Do not hesitate to assign "Bad" if an argument does not clearly meet the criteria for that dimension. The overall quality should reflect a synthesis of the other three dimensions but should also consider any other relevant factors. Do not assume that most arguments are "Good". Your priority is to identify weaknesses and be sensitive to any lack of quality. Return your response only as a JSON object using that values (Bad, Good). Do not use other labels. You MUST ONLY return a single JSON object with exactly these four fields: cogency, effectiveness, reasonableness, overall. Values MUST ONLY be "Good" or "Bad", wrapped in double quotes. DO NOT explain. DO NOT comment. DO NOT include any text before or after. ANY output not matching JSON format will be considered INVALID.###DIMENSIONS### 1. **Cogency (Justification Quality)** Evaluate only the justifications used to support the claim. Ask yourself: - Are the justifications believable and relevant to the author's point? - Do they provide enough support for the conclusion? 2. **Effectiveness (Persuasiveness and Presentation)** Assess how persuasive the presentation is. Ask yourself: - Is the author persuasive or credible? - Does the argument evoke emotions appropriately? - Is the language clear, appropriate and grammatically correct? - Is the argument logically ordered and easy to follow? 3. **Reasonableness (Contribution to Issue Resolution)** Consider the argument’s contribution to resolving the issue. Ask: - Would the target audience accept it? - Does it contribute meaningfully to the discussion? - Does it provide helpful information for reaching a conclusion? - Does it address counterarguments? 4. **Overall Quality** Reflect on the three dimensions above. Consider any other relevant factors for the general quality of the argument. ###ARGUMENT### {argument_text} ###OUTPUT### {json.dumps(output_json, ensure_ascii=False)}"""
    return text

# Procesa un dataset completo
def process_file(input_path, output_path):
    df = pd.read_csv(input_path)
    rows = []

    for _, row in df.iterrows():
        output = {
            "cogency": map_scores('cogency', row['cogency_mean']),
            "effectiveness": map_scores('effectiveness', row['effectiveness_mean']),
            "reasonableness": map_scores('reasonableness', row['reasonableness_mean']),
            "overall": map_scores('overall', row['overall_mean'])
        }
        full_prompt = build_prompt(row['text'], output)
        rows.append(full_prompt)

    output_df = pd.DataFrame({'text': rows})
    output_df.to_csv(output_path, index=False)
    print(f"✅ Saved {len(rows)} prompts to {output_path}")

# Procesar los tres splits
process_file('data/data_train.csv', 'data/ft_train.csv')
process_file('data/data_val.csv', 'data/ft_val.csv')
process_file('data/data_test.csv', 'data/ft_test.csv')

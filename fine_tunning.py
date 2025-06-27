import pandas as pd
import json

data_train = pd.read_csv('./data_train.csv')
data_val = pd.read_csv('./data_val.csv')
data_test = pd.read_csv('./data_test.csv')

def map_score(score):
    return 'Good' if score >= 3.33 else 'Bad'

def prepare_finetune_format(df):
    samples = []
    for _, row in df.iterrows():
        sample = {
            "input": row['text'],
            "output": f"Cogency: {map_score(row['cogency_mean'])}, Effectiveness: {map_score(row['effectiveness_mean'])}, Reasonableness: {map_score(row['reasonableness_mean'])}, Overall: {map_score(row['overall_mean'])}"
        }
        samples.append(sample)
    return samples

finetune_train = prepare_finetune_format(data_train)
finetune_val = prepare_finetune_format(data_val)
finetune_test = prepare_finetune_format(data_test)

with open('./finetune_train.json', 'w', encoding='utf-8') as f:
    for entry in finetune_train:
        f.write(json.dumps(entry, ensure_ascii=False) + '\n')

with open('./finetune_val.json', 'w') as f:
    for entry in finetune_val:
        f.write(json.dumps(entry,  ensure_ascii=False) + '\n')
    

with open('./finetune_test.json', 'w') as f:
    for entry in finetune_test:
        f.write(json.dumps(entry,  ensure_ascii=False) + '\n')

print("Fine-tuning JSON datasets created successfully.")
    
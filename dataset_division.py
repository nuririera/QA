import pandas as pd
from sklearn.model_selection import train_test_split

# Load the dataset
data = pd.read_csv('./dataset.csv')

# Select necesary columns
columns_of_interest = ['text', 'cogency_mean', 'effectiveness_mean', 'reasonableness_mean', 'overall_mean']
data = data[columns_of_interest]

# Map the cogency_mean, effectiveness_mean, reasonableness_mean, and overall_mean columns
def map_score(score):
    return 'Good' if score > 2.9 else 'Bad'

#Apply the map_score function to the cogency_mean, effectiveness_mean, reasonableness_mean, and overall_mean columns
for col in ['cogency_mean', 'effectiveness_mean', 'reasonableness_mean', 'overall_mean']:
    data[col] = data[col].apply(map_score)

seed = 42

# Split the data into train and test sets
data_train, data_temp = train_test_split(data, test_size=0.6, random_state=seed)
data_val, data_test = train_test_split(data_temp, test_size=0.07, random_state=seed)

def get_text_and_labels(df):
    return [
        {
            'text': row['text'],
            'labels': {
                'cogency': row['cogency_mean'],
                'effectiveness': row['effectiveness_mean'],
                'reasonableness': row['reasonableness_mean'],    
                'overall': row['overall_mean']
            }
        }
        for _, row in df.iterrows()
    ]

test_data = get_text_and_labels(data_test)


# Save the splits to CSV files
# data_train.to_csv('./data_train.csv', index=False)
# data_val.to_csv('./data_val.csv', index=False)
data_test.to_csv('./data_test.csv', index=False)
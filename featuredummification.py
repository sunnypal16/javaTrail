import pandas as pd

# Create the dataset
data = {
    'Country': ['France', 'Spain', 'Germany', 'Spain', 'Germany', 'France', 'Spain', 'France', 'Germany', 'France'],
    'Age': [44, 27, 30, 38, 40, 35, 31, 48, 50, 37],
    'Salary': [72000, 48000, 54000, 61000, 85000, 58000, 52000, 79000, 83000, 67000],
    'Purchased': ['No', 'Yes', 'No', 'No', 'Yes', 'Yes', 'No', 'Yes', 'No', 'Yes']
}

# Convert the dataset to a pandas DataFrame
df = pd.DataFrame(data)

# Perform one-hot encoding (feature dummification) on the 'Country' column
df = pd.get_dummies(df, columns=['Country'])

# Convert 'Purchased' column to numerical representation (1 for 'Yes', 0 for 'No')
df['Purchased'] = df['Purchased'].map({'Yes': 1, 'No': 0})

print(df)

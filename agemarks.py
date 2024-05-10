import pandas as pd

# Read the data from the CSV file into a DataFrame
df = pd.read_csv('agemarks.csv')

# Perform data pre-processing tasks
# Handling missing values
df.dropna(inplace=True)  # Drop rows with missing values

# Handling outliers
# For example, remove rows where 'Marks' column is an outlier
z_scores = (df['Marks'] - df['Marks'].mean()) / df['Marks'].std()
df = df[(z_scores < 3)]  # Keep rows where the z-score is less than 3

# Print the pre-processed DataFrame
print(df)

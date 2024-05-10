import pandas as pd

# Read the data from the CSV file into a DataFrame
df = pd.read_csv('bikesellpricebuypricee.csv')

# Perform transformations
# Display records of the bike having Buy Price greater than equal to 3000
print("Records of bikes with Buy Price greater than or equal to 3000:")
print(df[df['Buy Price'] >= 3000])

# Sort the bike data in ascending order
df_sorted = df.sort_values(by='Buy Price', ascending=True)
print("\nBike data sorted in ascending order of Buy Price:")
print(df_sorted)

# Group the data according to the "Model" of bike
grouped_data = df.groupby('Model')
print("\nGrouped data according to the Model of bike:")
for model, group in grouped_data:
    print("\nModel:", model)
    print(group)

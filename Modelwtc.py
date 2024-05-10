import pandas as pd

# Read data from CSV file into a DataFrame
car_data = pd.read_csv('car_data.csv')

# Display records of the car having Buy Price greater than or equal to 3000
print("Records with Buy Price >= 3000:")
print(car_data[car_data['Buy Price'] >= 3000])

# Sort the car data in ascending order
sorted_car_data = car_data.sort_values(by=['Buy Price'])

# Display the sorted DataFrame
print("Sorted Car Data:")
print(sorted_car_data)

# Group the data according to the “Model” of car
grouped_car_data = car_data.groupby('Model')

# Display the grouped data
print("Grouped Car Data:")
for model, group in grouped_car_data:
    print(model)
    print(group)
    print()

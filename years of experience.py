import numpy as np
from sklearn.linear_model import LinearRegression

# Data
X = np.array([2, 10, 4, 20, 8, 12, 22]).reshape(-1, 1)  # Years of Experience
y = np.array([30000, 95000, 45000, 178000, 84000, 120000, 200000])  # Salary

# Linear Regression Model
model = LinearRegression()

# Fit the model
model.fit(X, y)

# Predict for some years of experience
years_of_experience = [[5], [15]]  # You can add more values for prediction
predicted_salaries = model.predict(years_of_experience)

# Print the predicted salaries
for i, years in enumerate(years_of_experience):
    print("For {} years of experience, predicted salary is: ${:,.2f}".format(years[0], predicted_salaries[i]))

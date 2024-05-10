import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

# Load the dataset into a DataFrame
data = {
    "Alt": ["Yes", "Yes", "No", "Yes", "Yes", "No", "No", "No", "No"],
    "Bar": ["No", "No", "Yes", "No", "No", "Yes", "Yes", "No", "Yes"],
    "Fri": ["No", "No", "No", "Yes", "Yes", "No", "No", "No", "Yes"],
    "Hun": ["Yes", "Yes", "No", "Yes", "No", "Yes", "No", "Yes", "No"],
    "Pat": ["Some", "Full", "Some", "Full", "Full", "Some", "None", "Some", "Full"],
    "Price": [1200, 2500, 2200, 1245, 4300, 3400, 1000, 3200, 3400],
    "Rain": ["No", "No", "No", "No", "No", "Yes", "Yes", "Yes", "Yes"],
    "Res": ["Yes", "No", "No", "No", "Yes", "Yes", "No", "Yes", "No"],
    "Type": ["French", "Thai", "Burger", "Thai", "French", "Italian", "Burger", "Thai", "Burger"],
    "Est": ["0-10", "30-60", "0-10", "30-60", ">60", "0-10", "0-10", "0-10", ">60"],
    "Wait": ["Yes", "No", "Yes", "Yes", "No", "Yes", "No", "Yes", "No"]
}

df = pd.DataFrame(data)

# Convert categorical variables to numerical using LabelEncoder
label_encoders = {}
for column in df.columns:
    if df[column].dtype == 'object':
        label_encoders[column] = LabelEncoder()
        df[column] = label_encoders[column].fit_transform(df[column])

# Separate features (X) and target variable (y)
X = df.drop(columns=['Wait'])
y = df['Wait']

# Create and train the Decision Tree model
dt_model = DecisionTreeClassifier()
dt_model.fit(X, y)

# Visualize the decision tree
plt.figure(figsize=(20,10))
plot_tree(dt_model, feature_names=df.columns[:-1], class_names=['Not Waiting', 'Waiting'], filled=True)
plt.show()

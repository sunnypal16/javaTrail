import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt

# Load Titanic dataset (assuming it's in CSV format)
titanic_data = pd.read_csv("titanic.csv")

# Drop irrelevant columns or handle missing values as needed

# Convert categorical variables to numerical representation using one-hot encoding or label encoding

# Split dataset into features and target variable
X = titanic_data.drop(columns=['Survived'])  # Features
y = titanic_data['Survived']  # Target variable

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize the Decision Tree classifier
clf = DecisionTreeClassifier()

# Train the classifier on the training data
clf.fit(X_train, y_train)

# Plot the Decision Tree
plt.figure(figsize=(12, 8))
plot_tree(clf, filled=True, feature_names=X.columns, class_names=['Not Survived', 'Survived'])
plt.show()

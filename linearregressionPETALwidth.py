from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

iris = load_iris()
X = iris.data[:, 2:3]  
y = iris.data[:, 3]    

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

plt.scatter(X_train, y_train, color='blue', label='Training Data')

plt.scatter(X_test, y_test, color='red', label='Testing Data')

plt.plot(X_test, y_pred, color='green', linewidth=3, label='Regression Line')

plt.xlabel('Petal Length')
plt.ylabel('Petal Width')
plt.title('Linear Regression on Iris Dataset')
plt.legend()

plt.show()

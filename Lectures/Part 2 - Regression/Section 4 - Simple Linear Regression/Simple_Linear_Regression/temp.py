import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,1].values
#X is capital because its a matrix whereas y (the dependent variable) is 
#technically a vector


from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

t = regressor.predict(X_test)
#t is the target vector that predicts/estimates the value of y_test (the actual value) 

plt.scatter(X_train, y_train, c = 'firebrick')
plt.plot(X_train, regressor.predict(X_train), c = 'royalblue', alpha = 0.5)
plt.grid(True)
plt.title('Salary vs Years of Experience(Training Set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

plt.scatter(X_test, y_test, c = 'firebrick')
plt.plot(X_train, regressor.predict(X_train), c = 'royalblue', alpha = 0.5)
plt.grid(True)
plt.title('Salary vs Years of Experience')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()
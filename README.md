# Implementation of Multivariate Linear Regression
## Aim
To write a python program to implement multivariate linear regression and predict the output.
## Equipment’s required:
1.	Hardware – PCs
2.	Anaconda – Python 3.7 Installation / Moodle-Code Runner

## Algorithm:
1. Read the input dataset containing multiple input variables and the output variable.
2. Initialize the regression coefficients and learning rate
3. Calculate the predicted output using the linear regression equation.
4. Adjust the coefficients to minimize the prediction error.
5. Use the final model to predict the output for new data.

## Program:
```
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model, metrics
from sklearn.model_selection import train_test_split
housing = datasets.fetch_california_housing()
X = housing.data
y = housing.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=1)
reg = linear_model.LinearRegression()
reg.fit(X_train, y_train)
print('Coefficients: ', reg.coef_)
print('Variance score: {}'.format(reg.score(X_test, y_test)))
plt.style.use('fivethirtyeight')
plt.scatter(reg.predict(X_train), reg.predict(X_train) - y_train,
            color="green", s=10, label='Train data')
plt.scatter(reg.predict(X_test), reg.predict(X_test) - y_test,
            color="blue", s=10, label='Test data')
plt.hlines(y=0, xmin=0, xmax=5, linewidth=2)
plt.legend(loc='upper right')
plt.title('Residual Errors')
plt.show()
```

## Output:
<img width="805" height="606" alt="{B905A420-B0E4-4173-8CD1-77DCA62B7736}" src="https://github.com/user-attachments/assets/a00cefed-a7ae-4258-9727-94dbf4ab884c" />


## Result
Thus the multivariate linear regression is implemented and predicted the output using python program.

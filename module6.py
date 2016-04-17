# Module 6: Avoid overfitting with regularisation

# Previous imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from functions import PolynomialRegression
from sklearn.cross_validation import cross_val_score

# New imports
from functions import PolynomialRidge, PolynomialLasso
from sklearn.grid_search import GridSearchCV

# Load dataset
bikes_df = pd.read_csv('./data/bikes.csv')

# Code after this

features = ['temperature', 'humidity', 'windspeed']
x = bikes_df[features].values
y = bikes_df['count'].values

polynomial_ride = PolynomialLasso(degree = 2, alpha = 0)
polynomial_ride.fit(x,y)
coefs_0 = polynomial_ride.steps[2][1].coef_
print coefs_0

polynomial_ride = PolynomialLasso(degree=2, alpha=1)
polynomial_ride.fit(x,y)
coefs_1 = polynomial_ride.steps[2][1].coef_
print coefs_1

coefs = []
for alpha in np.logspace(-3,3,100):
    polynomial_ride = PolynomialLasso(degree = 2, alpha = alpha)
    polynomial_ride.fit(x,y)
    coefs.append(polynomial_ride.steps[2][1].coef_)
coefs = np.array(coefs)
plt.plot(np.logspace(-3,3,100),coefs)
plt.gca().set_xscale('log')
plt.show()

# Module 5: Evaluating model performance

# Previous imports

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from functions import PolynomialRegression

# New imports
from sklearn.cross_validation import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error

# Load dataset
bikes_df = pd.read_csv('./data/bikes_subsampled.csv')
temperature = bikes_df[['temperature']].values
bikes_count = bikes_df['count'].values

# Code after this
#lession 29
temp_train, temp_test, bikes_train, bikes_test = \
train_test_split(temperature, bikes_count, test_size=0.5)

#lession 30
polynomial_regression = PolynomialRegression(degree =3)
polynomial_regression.fit(temp_train, bikes_train)
temperature_predict = np.expand_dims(np.linspace(-5,40,50),1)
bikes_predict = polynomial_regression.predict(temperature_predict)
# plt.plot(temperature_predict, bikes_predict, linewidth = 2)
# plt.scatter(temp_train, bikes_train, color = 'k')
# plt.scatter(temp_test, bikes_test, color = 'r')
# plt.xlim(0,35)
# plt.ylim(0,1200)
# plt.show()

bikes_train_predict = polynomial_regression.predict(temp_train)
bikes_test_predict = polynomial_regression.predict(temp_test)

print 'Train MAE: ', mean_absolute_error(bikes_train, bikes_train_predict)
print 'Test MAE: ', mean_absolute_error(bikes_test, bikes_test_predict)

#lession 31

#plot the error vs the fitting

#lession 32
df = pd.read_csv('./data/bikes.csv')
temperature = df[['temperature']].values
bikes = df['count'].values
# plt.scatter(temperature, bikes, color = 'k')
# plt.show()

polynomial_regression = PolynomialRegression(degree =1)
scores = -cross_val_score(polynomial_regression, temperature, \
                          bikes, scoring='mean_absolute_error', cv = 5)

print scores

print np.mean(scores)


for degree in range(1,11):
    polynomial_regression = PolynomialRegression(degree = degree)
    score_cv = cross_val_score(polynomial_regression,temperature, \
                               bikes, scoring = 'mean_absolute_error', cv=5)
    score_cv_m = - np.mean(score_cv)
    plt.plot(degree, score_cv_m, 'bo')

plt.ylabel('Cross-validation score')
plt.xlabel('Polynomial degree')
plt.show()

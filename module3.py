# Module 3: Linear regression

# New imports
from Tkinter import Label
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn import metrics

# Code after this

bikes_df = pd.read_csv('data/bikes_subsampled.csv')
temperature = bikes_df['temperature'].values
bikes_count = bikes_df['count'].values

plt.scatter(temperature, bikes_count, color = 'k')
plt.xlabel('Temperature')
plt.ylabel('bikes hired')
plt.xlim(-5,40)
plt.tight_layout()

a = 28
temperature_predict = np.expand_dims(a=np.linspace(-5, 40, 100), axis=1)
bikes_count_predict = a * temperature_predict
plt.scatter(temperature, bikes_count, color = 'k')
plt.plot(temperature_predict, bikes_count_predict, linewidth = 2)

linear_regression = LinearRegression()
temperature_  = np.expand_dims(temperature,1)
linear_regression.fit(temperature_, bikes_count)

print 'bikes hired at 5 degree celsius: ', linear_regression.predict(5.)[0]
print 'optimal slope: ', linear_regression.coef_[0]
print 'optimal intercept: ', linear_regression.intercept_

plt.plot(temperature_predict, bikes_count_predict + linear_regression.intercept_, linewidth = 2)


bikes_count_predict = linear_regression.predict(temperature_)
print 'MAE: ', metrics.mean_absolute_error(bikes_count,bikes_count_predict)

def mape(y_true, y_pred):
    return 100*np.mean(np.abs(y_true-y_pred)/y_true)

print 'MAPE: ', mape(bikes_count, bikes_count_predict)

#plt.show()


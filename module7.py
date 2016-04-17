# Module 7: Predict the future with autoregression

# Previous imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

# New imports
from functions import organize_data

# Load dataset
bikes_df = pd.read_csv('./data/bikes.csv')
bikes = bikes_df['count'].values

# Code after this
k = 7
h = 1
x, y = organize_data(bikes, k, h)
# plt.plot(y, label='demand', color='k')
# plt.legend(loc=0)
# plt.xlabel('Days')
# plt.ylabel('Number of bikes hired')
m = 100
regressor = LinearRegression()
regressor.fit(x[:m],y[:m])
prediction = regressor.predict(x)

plt.plot(y, label='True Demand', color='k')
plt.plot(prediction, '-', color='b', label='Prediction')
plt.plot(y[:m], color='r', label='Train data')
plt.legend(loc=0)
plt.xlabel('Days')
plt.ylabel('Number of bikes hired')
plt.xlim(0,len(y))

plt.show()
print 'MAE: ', mean_absolute_error(y[m:], prediction[m:])


#lession 44
res = []
for k in range(1,40):
    x,y = organize_data(bikes,k,h)
    regressor.fit(x[:m],y[:m])
    yy = regressor.predict(x[m:])
    res.append(mean_absolute_error(y[m:],yy))

# plt.plot(res)
# plt.ylabel('Mean abs error')
# plt.show()

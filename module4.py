# Module 4: Multiple and polynomial regression

# Previous imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# New imports
from functions import PolynomialRegression
from mpl_toolkits.mplot3d.axes3d import Axes3D
from functions import model_plot_3d
from functions import polynomial_residual

# Load dataset
bikes_df = pd.read_csv('./data/bikes_subsampled.csv')

# Code after this

features = ['temperature', 'humidity']
x = bikes_df[features].values
y = bikes_df['count'].values

print x.shape, y.shape

# fig = plt.figure()
# ax = Axes3D(fig)
# ax.scatter(x[:,0], x[:,1],y)
# ax.set_label('the plot')
# ax.set_xlabel('temperature')
# ax.set_ylabel('humidity')
# ax.set_zlabel('bikes')

print 'correlation coef of temperature: ', \
    np.corrcoef(bikes_df['temperature'], bikes_df['count'])[0,1]

print 'correlation coef of humidity: ', \
    np.corrcoef(bikes_df['humidity'], bikes_df['count'])[0,1]

linear_regression = LinearRegression()
linear_regression.fit(x,y)


print 'Bikes hired at 20 degree and 60% humidity: ', \
    linear_regression.predict([20,60])


print 'Bikes hired at 5 degree and 90% humidity: ', \
    linear_regression.predict([5,90])


# fig = plt.figure()
# ax = Axes3D(fig)
# temperature_predict = np.linspace(0,35,50)
# humidity_predict = np.linspace(45,75,50)
#
# model_plot_3d(ax, linear_regression, temperature_predict, humidity_predict)
# ax.scatter(x[:,0],x[:,1], y)
#
# ax.set_label('the plot')
# ax.set_xlabel('temperature')
# ax.set_ylabel('humidity')
# ax.set_zlabel('bikes')


bikes_count = bikes_df['count'].values
temperature = bikes_df[['temperature']].values
polynomial_regression = PolynomialRegression(degree=5)
polynomial_regression.fit(temperature, bikes_count)
temperature_predict = np.expand_dims(np.linspace(-5,40,50),1)
bikes_count_predict = polynomial_regression.predict(temperature_predict)

#lession 26
# plt.scatter(temperature, bikes_count,color='k')
# plt.plot(temperature_predict,bikes_count_predict, linewidth=2)
# plt.legend(['Data', 'Degree 5'], loc = 0)
# plt.show()

#lession 27
degrees = range(1,15)
residual = []
for degree in degrees:
    residual.append(polynomial_residual(degree, temperature,bikes_count))

print residual
#lession 28
fig = plt.figure()
plt.plot(degrees, residual, 'bo--')
plt.ylabel('Residual (MAE)')
plt.xlabel('Polynomial degree')
plt.show()

import numpy as np 
from matplotlib import pyplot as plot
import pandas as pd

# h(x)= a+b(x)  the following function will find a,b where a=intercept, b= slope
#def find_slope_intercept(X_cordinate, Y_cordinate)


plot.rcParams['figure.figsize'] = (12.0, 9.0)

# Preprocessing Input data
data = pd.read_csv('data.csv')
X = data.iloc[:, 0]
Y = data.iloc[:, 1]
#plot.show()



Slope = 0
intercept = 0

alpha = 0.0001  # The learning Rate
interations = 12500  # The number of iterations to perform gradient descent


n = float(len(X)) # Number of elements in X

# Performing Gradient Descent 
for i in range(interations): 
    Y_pred = Slope*X + intercept  # The current predicted value of Y
    Deri_Slope = (-1/n) * sum(X * (Y - Y_pred))  # Derivative wrt m
    Deri_Intercept = (-1/n) * sum(Y - Y_pred)  # Derivative wrt c
    Slope = Slope - alpha * Deri_Slope  # Update m
    intercept = intercept - alpha * Deri_Intercept  # Update c
    
print (Slope, intercept)



plot.scatter(X, Y) 
plot.plot([min(X), max(X)], [min(Y_pred), max(Y_pred)], color='red')  # regression line

plot.show()
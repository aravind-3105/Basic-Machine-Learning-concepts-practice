import numpy as np 
from sklearn.linear_model import LinearRegression
from matplotlib import pyplot as plot
import array


X_cordinate = np.array([5,15,25,35,45,55]).reshape((-1,1))
Y_coodinate = np.array([54, 20, 14, 32, 22, 38])
plot.title("Linear Regression") 
plot.xlabel("X-Axis") 
plot.ylabel("Y-Axis") 
plot.scatter(X_cordinate, Y_coodinate, label='Data Set') 

""" 
print(X_cordinate)
print("\n")
print(Y_coodinate)
This sort of vectors are produced to be sent as inputs to 
[[ 5]
 [15]
 [25]
 [35]
 [45]
 [55]]


[ 5 20 14 32 22 38]

"""

model = LinearRegression()     #inbuilt function as a part of the scikit-learn package
model.fit(X_cordinate, Y_coodinate)


print('Intercept:', model.intercept_)
print('Slope:', model.coef_)


x = np.linspace(-5,60,100)
abline_values = [model.coef_ * i + model.intercept_ for i in x]
# Plot the best fit line over the actual values
plot.plot(x, abline_values, ':b')
plot.show()




print("Number of values to predict")
size = int(input())
#print(size)
#To predict a response 
a = []
count = 0
while count < size:        
	print("Enter the X-cordinate value")
	X = int(input())
	a.append(X)
	count+=1

to_predict=np.asarray(a).reshape((-1,1))

predicted_Y_cordinate = model.predict(to_predict)
plt.plot(to_predict , predicted_Y_cordinate, label='Line obtained from linear Regression')
print("\n")
print('predicted response:', predicted_Y_cordinate)
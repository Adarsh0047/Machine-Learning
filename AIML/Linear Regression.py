import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


#reading file into object
dataset=pd.read_csv('D:\\Old pc\\Python\\Python Lab\\AIML\\age_weight.csv')


#give x and y values of attribute(Independent) and label(Value to be predicted-Dependent)
X=dataset.iloc[:,:-1].values#[rows,column] calling 0th column. Notice  ':' in this line
y=dataset.iloc[:,1].values#[rows,column] calling 1st column. Notice there is no ':' in this line


#train the model
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2, random_state=0)


#Fit model to our dataset and print intercept,slope,Actual and predicted values
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,y_train)
print('Intercept:',regressor.intercept_)
print('Slope:',regressor.coef_)
y_pred=regressor.predict(X_test)
df=pd.DataFrame({'Actual': y_test,'Predicted':y_pred})
print(df)


#print mean absolute error,mean squared error and root mean squared error
from sklearn import metrics
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test,y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test,y_pred))
print('Root Mean Squared Error:',np.sqrt(metrics.mean_squared_error(y_test,y_pred)))


#plot our data(training set)
plt.scatter(X_train, y_train,color ="green")
plt.plot(X_train, regressor.predict(X_train),color="red")
plt.title("Weight vs Age(Training set)")
plt.xlabel("Age")
plt.ylabel("Weight")
plt.show()


#plot our data(test set)
plt.scatter(X_test, y_test,color ="green")
plt.plot(X_train, regressor.predict(X_train),color="red")
plt.title("Weight vs Age(Testing set)")
plt.xlabel("Age")
plt.ylabel("Weight")
plt.show()





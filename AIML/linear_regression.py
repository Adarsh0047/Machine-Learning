import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset=pd.read_csv('D:\\Old pc\\Python\\Python Lab\\AIML\\age_weight.csv')

x=dataset.iloc[:,:-1].values
y=dataset.iloc[:,1].values

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(x_train,y_train)
print('Intercept:',regressor.intercept_)
print('Coefficient:',regressor.coef_)
y_pred=regressor.predict(x_test)
df=pd.DataFrame({'Actual':y_test,'Predicted':y_pred})
print(df)

from sklearn import metrics
print('Mean Absolute Error:',metrics.mean_absolute_error(y_test,y_pred))
print('Mean Squared Error:',metrics.mean_squared_error(y_test,y_pred))
print('Root Mean Squared Error:',np.sqrt(metrics.mean_squared_error(y_test,y_pred)))

plt.scatter(x_train, y_train,color ="green")
plt.plot(x_train, regressor.predict(x_train),color="red")
plt.title("Weight vs Age(Training set)")
plt.xlabel("Age")
plt.ylabel("Weight")
plt.show()

plt.scatter(x_test, y_test,color ="green")
plt.plot(x_train, regressor.predict(x_train),color="red")
plt.title("Weight vs Age(Testing set)")
plt.xlabel("Age")
plt.ylabel("Weight")
plt.show()

print(regressor.predict([[18]]))

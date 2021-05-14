import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
dataset= pd.read_csv('D:\\Old pc\\Python\\Python Lab\\AIML\\heart.csv')

x=dataset.iloc[:,[4,7,8]].values
y=dataset.iloc[:,-1].values

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=0)

from sklearn.preprocessing import StandardScaler
sc_x=StandardScaler()
x_train=sc_x.fit_transform(x_train)
x_test=sc_x.transform(x_test)
print(x_train[0:10,:])

from sklearn.svm import SVC
classifier=SVC(kernel='linear')
classifier.fit(x_train,y_train)
print('Intercept:',classifier.intercept_)
print('Slope:',classifier.coef_)
y_pred=classifier.predict(x_test)
df=pd.DataFrame({'Actual': y_test,'Predicted':y_pred})
print(df)

from sklearn.metrics import accuracy_score
print ("Accuracy : ", accuracy_score(y_test, y_pred))



from sklearn import metrics
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test,y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test,y_pred))
print('Root Mean Squared Error:',np.sqrt(metrics.mean_squared_error(y_test,y_pred)))



print(classifier.predict([[190,160,1]]))


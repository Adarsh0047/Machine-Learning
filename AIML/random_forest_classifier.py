import pandas as pd
import numpy as np

dataframe=pd.read_csv('D:\\Old pc\\Python\\Python Lab\\AIML\\random_forest.csv')
x=dataframe.iloc[:,[0,1]].values
y=dataframe.iloc[:,2].values

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=0)

from sklearn.ensemble import RandomForestClassifier
tree=RandomForestClassifier(n_estimators=100)
tree.fit(x_train,y_train)

y_pred = tree.predict(x_test)

from sklearn import metrics
print("ACCURACY OF THE MODEL: ", metrics.accuracy_score(y_test, y_pred))

print(tree.predict([[18,55]]))

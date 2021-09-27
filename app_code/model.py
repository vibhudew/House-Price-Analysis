import pandas as pd 
import numpy as np 
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


df=pd.read_csv("kc_house_data.csv")

#the id,date, column no need for analysis so drop it
df1=df.drop(['id','date','yr_built','yr_renovated', 'zipcode'],axis=1)

cdf = df1[['bedrooms','bathrooms','sqft_living','sqft_lot','sqft_above','sqft_basement','floors']]

x = cdf.iloc[:, :6]
y = cdf.iloc[:, -1]


regressor = LinearRegression()
regressor.fit(x, y)


X_train,X_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=5)

clf = regressor.fit(X_train,y_train)
y_pred = clf.predict(X_test)


ID=np.array(np.round(y_test))
Prediction=np.array(np.round(y_pred))


accuracy_test = accuracy_score(ID,Prediction)
print(accuracy_test)

import pickle
pickle.dump(regressor,open('regressor.pkl','wb'))

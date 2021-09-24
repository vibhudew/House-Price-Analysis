import pandas as pd 
import numpy as np 
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

df=pd.read_csv("kc_house_data.csv")

#the id,date, column no need for analysis so drop it
df1=df.drop(['id','date','yr_built','yr_renovated', 'zipcode'],axis=1)

# df1.head(5)

# df1.describe()

# cdf = df1[['bedrooms','bathrooms','floors','sqft_living','sqft_lot','sqft_above','sqft_basement','price']]
cdf = df1[['bedrooms','bathrooms','sqft_living','sqft_lot','sqft_above','sqft_basement','floors']]

x = cdf.iloc[:, :6]
y = cdf.iloc[:, -1]
# print(y)




# kmean=KMeans(n_jobs = -1, n_clusters = 6, init='k-means++')
# kmean.fit(x, y)

regressor = LinearRegression()
regressor.fit(x, y)



import pickle
pickle.dump(regressor,open('sample61134678.pkl','wb'))



# Loading model to compare the results
model = pickle.load(open('sample6113467.pkl','rb'))
print(model.predict([[3,1,1180,5650,1180,0,1]]))


# # standardizing the data
# from sklearn.preprocessing import StandardScaler
# scaler = StandardScaler()
# data_scaled = scaler.fit_transform(df1)

# # statistics of scaled data
# pd.DataFrame(data_scaled).describe()

# # defining the kmeans function with initialization as k-means++
# kmeans = KMeans(n_clusters=3, init='k-means++')

# # fitting the k means algorithm on scaled data
# kmeans.fit(data_scaled)

# # inertia on the fitted data
# kmeans.inertia_

# # fitting multiple k-means algorithms and storing the values in an empty list
# SSE = []
# for cluster in range(1,20):
#     kmeans = KMeans(n_jobs = -1, n_clusters = cluster, init='k-means++')
#     kmeans.fit(data_scaled)
#     SSE.append(kmeans.inertia_)

# # converting the results into a dataframe and plotting them
# frame = pd.DataFrame({'Cluster':range(1,20), 'SSE':SSE})
# plt.figure(figsize=(12,6))
# plt.plot(frame['Cluster'], frame['SSE'], marker='o')
# plt.xlabel('Number of clusters')
# plt.ylabel('Inertia')

# # k means using 6 clusters and k-means++ initialization
# kmeans = KMeans(n_jobs = -1, n_clusters = 2, init='k-means++')
# sv=kmeans.fit(data_scaled)
# pred = kmeans.predict(data_scaled)


# frame = pd.DataFrame(data_scaled)
# frame['cluster'] = pred
# frame['cluster'].value_counts()

# cluster1=kmeans()

# value=frame['cluster'].value_counts()
# print(value)

# import pickle
# pickle.dump(sv,open('sample6.pkl','wb'))

#downloading the csv file which contaning the ID and the predicted values
# from pandas import DataFrame

# ID=np.array(df['id'])
# Prediction=np.array(pred)

# val=np.hstack((ID[:,np.newaxis],Prediction[:,np.newaxis]))
# val

# df = DataFrame(val, columns= ['',''])
# export_csv = df.to_csv (r'F:\cluster_house123.csv', index = None, header=True)
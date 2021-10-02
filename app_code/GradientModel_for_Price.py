import pandas as pd
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
#get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
import scipy.stats as stats
pd.set_option('float_format', '{:f}'.format)

df = pd.read_csv('kc_cleaned_for_pricePredict.csv') 

df.tail()

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn import model_selection

df = df[['bedrooms', 'bathrooms', 'floors', 'waterfront', 'view',
       'condition', 'grade', 'yr_built', 'yr_renovated', 'lat', 'long',
      'sqft_total','price']]

X = df.iloc[:, :12]
y = df.iloc[:, -1]

X_train, X_test, y_train, y_test = train_test_split(X, y,  test_size=.15, random_state=170378)
X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train,  test_size=.18, random_state=170378)

print(f"Train Data Shape: {X_train.shape}")
print(f"Valid Data Shape: {X_valid.shape}")
print(f"Test Data Shape: {X_test.shape}")

#Modeling

from xgboost import XGBRegressor
from sklearn.ensemble import GradientBoostingRegressor #GBR algorithm
import matplotlib.pylab as plt
from sklearn.model_selection import cross_validate

#get_ipython().run_line_magic('matplotlib', 'inline')
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 12, 4
from sklearn.metrics import mean_squared_error, r2_score

from sklearn import set_config
set_config(display='diagram',)

#A DataFRame to store the results is created:

index = ['GBRegressor']
col = ['R2 Train', 'RMSE Train','R2 Valid', 'RMSE Valid']

results_df_log = pd.DataFrame(index=index, columns=col)
results_df_lev = pd.DataFrame(index=index, columns=col)

learn_rate = 0.03 #g_best_model['m'].best_params_.get('learning_rate')
n_est = 700 #g_best_model['m'].best_params_.get('n_estimators')
tree_md = 8 #g_best_model['m'].best_params_.get('max_depth')


from sklearn.model_selection import cross_val_score

# Various hyper-parameters to tune

gradientBR = GradientBoostingRegressor(learning_rate=learn_rate,n_estimators=n_est,max_depth=8,min_samples_split=1200
                                    ,min_samples_leaf=60,subsample=0.85,random_state=10,max_features=7)

#Use K Fold cross validation to measure accuracy of our GBRegression model
print(cross_val_score(gradientBR,X_train, y_train,cv=5))

gradientBR_model = gradientBR.fit(X_train, y_train)
gradientBR_model

#Train Score: 
print(f'Score on Training set: {gradientBR_model.score(X_train, y_train)}')

#Validation Score:
print(f'Score on Valuation set: {gradientBR_model.score(X_valid, y_valid)}')

y_hat_train = gradientBR_model.predict(X_train)
y_hat_valid = gradientBR_model.predict(X_valid)


#fig, ax = plt.subplots(1,2,figsize=(11,5), sharey=True, sharex=True)

#ax[0].scatter(y_hat_train,y_train)
#ax[1].scatter(y_hat_valid,y_valid, c='r')
#ax[0].set_title('Train Dataset')
#ax[1].set_title('Validation Dataset')

#plt.suptitle('GBRegressor');

mse_train_xgb = mean_squared_error(y_train, y_hat_train, squared=False)
mse_valid_xgb = mean_squared_error(y_valid, y_hat_valid, squared=False)

r2_train_xgb = r2_score(y_train, y_hat_train)
r2_valid_xgb = r2_score(y_valid, y_hat_valid)

print(f'MSE Score on Training set: {mse_train_xgb}')
print(f'MSE Score on Validation set: {mse_valid_xgb}')
print('\n')
print(f'R2 Score on Training set: {r2_train_xgb}')
print(f'R2 Score on Validation set: {r2_valid_xgb}')


results_df_log.loc['XGBRegressor','R2 Train'] = r2_train_xgb
results_df_log.loc['XGBRegressor','R2 Valid'] = r2_valid_xgb
results_df_log.loc['XGBRegressor','RMSE Train'] = mse_train_xgb
results_df_log.loc['XGBRegressor','RMSE Valid'] = mse_valid_xgb

#GBRagressor is the model delivering the best results on the validation dataset. 
#The table below summarizes the overall results using the target feature in log:

results_df_log

#The model is now tested on the test dataset

y_hat_test = gradientBR_model.predict(X_test)


mse_test_xgb = mean_squared_error(y_test, y_hat_test, squared=False)
r2_test_xgb = r2_score(y_test, y_hat_test)

print(f'MSE Score on Test set: {mse_test_xgb}')
print('\n')
print(f'R2 Score on Test set: {r2_test_xgb}')

import matplotlib.pyplot as plt

x= y_hat_test
y = y_test
plt.scatter(x,y)
plt.title("Real  Vs Prediction")
plt.xlabel("Predicted Price")
plt.ylabel("Real Price")
#plt.show()


prediction = pd.DataFrame(index=y_test.index,columns=['Real Value','Prediction','Difference'])

prediction['Real Value'] = np.round(y_test,0)
prediction['Prediction'] = np.round(y_hat_test,0)
prediction['Difference'] = np.round(abs(y_hat_test - y_test),0)

#Convert DataFrame to a csv file that can be uploaded
#This is saved in the same directory as your notebook
filename = 'House Price Prediction.csv'

prediction.to_csv(filename,index=False)

print('Saved file: ' + filename)


prediction.dtypes


prediction.Prediction = prediction.Prediction.astype('int64')
prediction.Difference = prediction.Difference.astype('int64')


print(prediction)


prediction.head(30)


#calculate model score
print("Gradient Boosting Regression Model Score is ",round(gradientBR_model.score(X_test,y_test)*100))


import pickle
with open('gradientModel.pickle','wb') as f:
    pickle.dump(gradientBR_model,f)


df.columns


df.dtypes




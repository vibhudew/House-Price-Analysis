
import pandas as pd
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
##get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
import scipy.stats as stats
#pd.set_option('float_format', '{:f}'.format)

df = pd.read_csv(r'C:\Users\singa\Desktop\New folder\House-Price-Analysis\app_code\kc_house_data.csv') 

df.tail()


df.info()


#correcting data types
##discrete vars as int
#df.price = df.price.astype('int64')
df.bedrooms = df.bedrooms.astype('int64')
#df.bathrooms = df.bathrooms.astype('float64')
df.floors = df.floors.astype('int64')
df.grade = df.grade.astype('int64')

#df.lat = df.lat.astype('int64')
#df.long = df.long.astype('int64')
df.sqft_living = df.sqft_living.astype('int64')
df.sqft_lot = df.sqft_lot.astype('int64')
df.sqft_above = df.sqft_above.astype('int64')
df.sqft_basement = df.sqft_basement.astype('int64')


#drop unnecessary column
df = df.drop(['id','date'],axis=1)

df.describe(include='all')


#drop unnecessary column
df = df.drop(['sqft_living15','sqft_lot15'],axis=1)


df.isnull().sum()


df.shape

df.duplicated()


df.drop_duplicates(keep=False,inplace=True)

df.duplicated().sum()


## Adjustments
f,ax=plt.subplots(figsize=(15, 20),nrows=5,ncols=2)
plt.subplots_adjust(hspace=1)
## Features
sns.boxplot(df['price'],data=df,ax=ax[0][0])
sns.boxplot(df['bedrooms'],data=df,ax=ax[0][1])
sns.boxplot(df['bathrooms'],data=df,ax=ax[1][0])
sns.boxplot(df['sqft_living'],data=df,ax=ax[1][1])
sns.boxplot(df['floors'],data=df,ax=ax[2][0])
sns.boxplot(df['grade'],data=df,ax=ax[2][1])
sns.boxplot(df['sqft_above'],data=df,ax=ax[3][0])
sns.boxplot(df['sqft_basement'],data=df,ax=ax[3][1])
sns.boxplot(df['lat'],data=df,ax=ax[4][0])


#price

#Q1 & Q2 defination
Q1 = df['price'].quantile(0.25)
Q3 = df['price'].quantile(0.75)
print('Q1:',Q1)
print('Q3: ',Q3)

IQR = Q3-Q1
print('IQR: ',IQR)

lower_limit = Q1 - 1.5*IQR
upper_limit = Q3 + 1.5*IQR
print('Lower limit: ',lower_limit)
print('Upper limit: ',upper_limit)

df['price'] = np.where(df['price']>upper_limit,upper_limit,df['price'])
df['price'] = np.where(df['price']<lower_limit,lower_limit,df['price'])


#bedrooms

#Q1 & Q2 defination
Q1 = df['bedrooms'].quantile(0.25)
Q3 = df['bedrooms'].quantile(0.75)
print('Q1:',Q1)
print('Q3: ',Q3)

IQR = Q3-Q1
print('IQR: ',IQR)

lower_limit = Q1 - 1.5*IQR
upper_limit = Q3 + 1.5*IQR
print('Lower limit: ',lower_limit)
print('Upper limit: ',upper_limit)

df['bedrooms'] = np.where(df['bedrooms']>upper_limit,upper_limit,df['bedrooms'])
df['bedrooms'] = np.where(df['bedrooms']<lower_limit,lower_limit,df['bedrooms'])


#bathrooms

#Q1 & Q2 defination
Q1 = df['bathrooms'].quantile(0.25)
Q3 = df['bathrooms'].quantile(0.75)
print('Q1:',Q1)
print('Q3: ',Q3)

IQR = Q3-Q1
print('IQR: ',IQR)

lower_limit = Q1 - 1.5*IQR
upper_limit = Q3 + 1.5*IQR
print('Lower limit: ',lower_limit)
print('Upper limit: ',upper_limit)

df['bathrooms'] = np.where(df['bathrooms']>upper_limit,upper_limit,df['bathrooms'])
df['bathrooms'] = np.where(df['bathrooms']<lower_limit,lower_limit,df['bathrooms'])


#sqft_living

#Q1 & Q2 defination
Q1 = df['sqft_living'].quantile(0.25)
Q3 = df['sqft_living'].quantile(0.75)
print('Q1:',Q1)
print('Q3: ',Q3)

IQR = Q3-Q1
print('IQR: ',IQR)

lower_limit = Q1 - 1.5*IQR
upper_limit = Q3 + 1.5*IQR
print('Lower limit: ',lower_limit)
print('Upper limit: ',upper_limit)

df['sqft_living'] = np.where(df['sqft_living']>upper_limit,upper_limit,df['sqft_living'])
df['sqft_living'] = np.where(df['sqft_living']<lower_limit,lower_limit,df['sqft_living'])


#floors

#Q1 & Q2 defination
Q1 = df['floors'].quantile(0.25)
Q3 = df['floors'].quantile(0.75)
print('Q1:',Q1)
print('Q3: ',Q3)

IQR = Q3-Q1
print('IQR: ',IQR)

lower_limit = Q1 - 1.5*IQR
upper_limit = Q3 + 1.5*IQR
print('Lower limit: ',lower_limit)
print('Upper limit: ',upper_limit)

df['floors'] = np.where(df['floors']>upper_limit,upper_limit,df['floors'])
df['floors'] = np.where(df['floors']<lower_limit,lower_limit,df['floors'])

#grade

#Q1 & Q2 defination
Q1 = df['grade'].quantile(0.25)
Q3 = df['grade'].quantile(0.75)
print('Q1:',Q1)
print('Q3: ',Q3)

IQR = Q3-Q1
print('IQR: ',IQR)

lower_limit = Q1 - 1.5*IQR
upper_limit = Q3 + 1.5*IQR
print('Lower limit: ',lower_limit)
print('Upper limit: ',upper_limit)

df['grade'] = np.where(df['grade']>upper_limit,upper_limit,df['grade'])
df['grade'] = np.where(df['grade']<lower_limit,lower_limit,df['grade'])



#sqft_above

#Q1 & Q2 defination
Q1 = df['sqft_above'].quantile(0.25)
Q3 = df['sqft_above'].quantile(0.75)
print('Q1:',Q1)
print('Q3: ',Q3)

IQR = Q3-Q1
print('IQR: ',IQR)

lower_limit = Q1 - 1.5*IQR
upper_limit = Q3 + 1.5*IQR
print('Lower limit: ',lower_limit)
print('Upper limit: ',upper_limit)

df['sqft_above'] = np.where(df['sqft_above']>upper_limit,upper_limit,df['sqft_above'])
df['sqft_above'] = np.where(df['sqft_above']<lower_limit,lower_limit,df['sqft_above'])


#sqft_basement

#Q1 & Q2 defination
Q1 = df['sqft_basement'].quantile(0.25)
Q3 = df['sqft_basement'].quantile(0.75)
print('Q1:',Q1)
print('Q3: ',Q3)

IQR = Q3-Q1
print('IQR: ',IQR)

lower_limit = Q1 - 1.5*IQR
upper_limit = Q3 + 1.5*IQR
print('Lower limit: ',lower_limit)
print('Upper limit: ',upper_limit)

df['sqft_basement'] = np.where(df['sqft_basement']>upper_limit,upper_limit,df['sqft_basement'])
df['sqft_basement'] = np.where(df['sqft_basement']<lower_limit,lower_limit,df['sqft_basement'])


#lat

#Q1 & Q2 defination
Q1 = df['lat'].quantile(0.25)
Q3 = df['lat'].quantile(0.75)
print('Q1:',Q1)
print('Q3: ',Q3)

IQR = Q3-Q1
print('IQR: ',IQR)

lower_limit = Q1 - 1.5*IQR
upper_limit = Q3 + 1.5*IQR
print('Lower limit: ',lower_limit)
print('Upper limit: ',upper_limit)

df['lat'] = np.where(df['lat']>upper_limit,upper_limit,df['lat'])
df['lat'] = np.where(df['lat']<lower_limit,lower_limit,df['lat'])



## Adjustments
f,ax=plt.subplots(figsize=(20, 20),nrows=5,ncols=2)
plt.subplots_adjust(hspace=1)
## Features
sns.boxplot(df['price'],data=df,ax=ax[0][0])
sns.boxplot(df['bedrooms'],data=df,ax=ax[0][1])

sns.boxplot(df['bathrooms'],data=df,ax=ax[1][0])
sns.boxplot(df['sqft_living'],data=df,ax=ax[1][1])

sns.boxplot(df['floors'],data=df,ax=ax[2][0])
sns.boxplot(df['grade'],data=df,ax=ax[2][1])

sns.boxplot(df['sqft_above'],data=df,ax=ax[3][0])
sns.boxplot(df['sqft_basement'],data=df,ax=ax[3][1])

sns.boxplot(df['lat'],data=df,ax=ax[4][0])



corr = df.corr()
plt.figure(figsize=(20,16))
sns.heatmap(data=corr, square=True , annot=True, cbar=True)
plt.show()


price_corr = df.corr()['price'].sort_values(ascending=False)
print(price_corr)


#Feature Analysis: Bedrooms, Floors and Bathrooms:

f, axes = plt.subplots(1, 2,figsize=(15,5))
sns.boxplot(x=df['bedrooms'],y=df['price'], ax=axes[0], palette = 'autumn_r')
sns.boxplot(x=df['floors'],y=df['price'], ax=axes[1], palette = 'autumn_r')
sns.despine(left=True, bottom=True)
axes[0].set(xlabel='Bedrooms', ylabel='Price')
axes[0].yaxis.tick_left()
axes[1].yaxis.set_label_position("right")
axes[1].yaxis.tick_right()
axes[1].set(xlabel='Floors', ylabel='Price')

f, axe = plt.subplots(1, 1,figsize=(15,5))
sns.despine(left=True, bottom=True)
sns.boxplot(x=df['bathrooms'],y=df['price'], ax=axe, palette = 'autumn_r')
axe.yaxis.tick_left()
axe.set(xlabel='Bathrooms', ylabel='Price');



df['sqft_total'] = df['sqft_basement'] + df['sqft_living']+df['sqft_lot']+df['sqft_above']
df.sqft_total = df.sqft_total.astype('int64')



df.columns


df = df.drop(['sqft_living','sqft_lot','sqft_above','sqft_basement','zipcode'],axis=1)



df.dtypes



df.columns



cols_to_plot = df[['price', 'bedrooms', 'bathrooms', 'floors', 'waterfront', 'view',
       'condition', 'grade', 'yr_built', 'yr_renovated', 'lat', 'long',
       'sqft_total']]


# Function to plot scatterplots

def plot_scatterplots():
    for i in cols_to_plot.columns:
        cat_num = cols_to_plot[i].value_counts()
        print('Graph for {}: Total = {}'.format(i.capitalize(), len(cat_num)))
        sns.scatterplot(x=cat_num.index, y=cat_num)
        plt.xticks(rotation=90)
        plt.show()
        
plot_scatterplots()


df.isna().sum()



df.dtypes


df.columns



from sklearn.model_selection import train_test_split, GridSearchCV

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

from sklearn.metrics import mean_squared_error, r2_score

from sklearn import set_config
set_config(display='diagram',)


#A DataFRame to store the results is created:

index = ['XGBRegressor']
col = ['R2 Train', 'RMSE Train','R2 Valid', 'RMSE Valid']

results_df_log = pd.DataFrame(index=index, columns=col)
results_df_lev = pd.DataFrame(index=index, columns=col)


learn_rate = 0.03 #xg_best_model['m'].best_params_.get('learning_rate')
n_est = 700 #xg_best_model['m'].best_params_.get('n_estimators')
tree_md = 8 #xg_best_model['m'].best_params_.get('max_depth')


from sklearn.model_selection import cross_val_score
# Various hyper-parameters to tune
xgb_opt = XGBRegressor(learning_rate=learn_rate,
                       n_estimators=n_est,
                       max_depth=tree_md,
                       nthread=4,
                       subsample=0.9,
                       colsample_bytree=0.7,
                       min_child_weight=4,
                       objective='reg:squarederror')



#cross value score

print(cross_val_score(xgb_opt,X_train, y_train,cv=5))

best_xg_model = xgb_opt.fit(X_train, y_train)
best_xg_model

#Use K Fold cross validation to measure accuracy of our XGBRegression model

from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import cross_val_score

cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=0)
cross_val_score(XGBRegressor(), X, y, cv=cv)


#Train Score: 
print(f'Score on Training set: {best_xg_model.score(X_train, y_train)}')

#Validation Score:
print(f'Score on Valuation set: {best_xg_model.score(X_valid, y_valid)}')


y_hat_train = best_xg_model.predict(X_train)
y_hat_valid = best_xg_model.predict(X_valid)

#y_hat_train_lev = np.exp(y_hat_train)
#y_hat_valid_lev = np.exp(y_hat_valid)


fig, ax = plt.subplots(1,2,figsize=(11,5), sharey=True, sharex=True)

ax[0].scatter(y_hat_train,y_train)
ax[1].scatter(y_hat_valid,y_valid, c='r')
ax[0].set_title('Train Dataset')
ax[1].set_title('Validation Dataset')

plt.suptitle('XGBRegressor');


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


#XGBRagressor is the model delivering the best results on the validation dataset. 
#The table below summarizes the overall results using the target feature in log:

results_df_log


#The model is now tested on the test dataset

y_hat_test = best_xg_model.predict(X_test)


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
plt.show()


prediction = pd.DataFrame(index=y_test.index,columns=['Real Value','Prediction','Difference'])

prediction['Real Value'] = np.round(y_test,0)
prediction['Prediction'] = np.round(y_hat_test,0)
prediction['Difference'] = np.round(abs(y_hat_test - y_test),0)

#Convert DataFrame to a csv file that can be uploaded
#This is saved in the same directory as your notebook
filename = 'King County House Prediction.csv'

prediction.to_csv(filename,index=False)

print('Saved file: ' + filename)


price_pred= prediction['Prediction']
price_diff = prediction['Difference']
df_test=pd.DataFrame({'price_actual':prediction['Real Value'],'price_predicted':price_pred ,'difference' :price_diff})
df_test.head(30)


df.columns


#calculate model score
print("XGBRegression Model Score is ",round(xgb_opt.score(X_test,y_test)*100))


import pickle
with open('xgbmodel.pkl','wb') as f:
    pickle.dump(xgb_opt,f)



df.columns


df.dtypes


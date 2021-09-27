# data analysis and wrangling
import pandas as pd
import numpy as np
import random as rnd

# visualization
import seaborn as sns
import matplotlib.pyplot as plt
# %matplotlib inline

df = pd.read_csv(r'C:\Users\singa\FDM Mini\House-Price-Analysis\app_code\kc_house_data.csv')


print(df.columns.values)

# preview the data
df.head()

# No missing values
df.isnull().sum()

df.info()


df.describe().transpose()

#Cleaning

#changing question marks to 0.0
df = df.replace('?',0.0)
df.view = df.view.replace(np.nan,0)

#changing all column object types to floats (except date column)
df.loc[:, df.columns != 'date'] = df.loc[:,df.columns != 'date'].astype('float')

#changing all 0.0 in sqft_basement column, yr_renovated, and waterfront columns to NaN values
df['sqft_basement'] = df['sqft_basement'].replace(0.0 , np.nan)
df['waterfront'] = df.waterfront.replace(0.0, np.nan)
df['yr_renovated'] =df['yr_renovated'].replace(0.0, np.nan)

#changing date column to datetime values
df['date'] = pd.to_datetime(df['date'])
df['yr_sold'] = df['date'].dt.to_period('Y')


#Feature Engineering

#creating eff_built column (which updates built year depending on whether it was renovated or not)
#df.loc[df['yr_renovated'].notnull(), 'eff_built'] = 2022 - df['yr_renovated']
#df.loc[df['yr_renovated'].isnull(), 'eff_built'] = 2022 -df['yr_built']
#df.eff_built = df.eff_built.astype('int64')

#correcting data types
##discrete vars as int
df.bedrooms = df.bedrooms.astype('int64')
df.bathrooms = df.bathrooms.astype('int64')
df.floors = df.floors.astype('int64')
df.condition = df.condition.astype('int64')
df.grade = df.grade.astype('int64')
df.view = df.view.astype('int64')

#drop pre-processed columns
df = df.drop(['id','date','zipcode','sqft_living15','sqft_lot15'],axis=1)


# 
# ## Exploratory Data Analysis

# ### Pearson correlation matrix

sns.set(style="whitegrid", font_scale=1)

plt.figure(figsize=(13,13))
plt.title('Pearson Correlation Matrix',fontsize=25)
sns.heatmap(df.corr(),linewidths=0.25,vmax=0.7,square=True,cmap="GnBu",linecolor='w',
            annot=True, annot_kws={"size":7}, cbar_kws={"shrink": .7})


price_corr = df.corr()['price'].sort_values(ascending=False)
print(price_corr)


#f, axes = plt.subplots(1, 2,figsize=(15,5))
#sns.distplot(df['price'], ax=axes[0])
#sns.scatterplot(x='price',y='sqft_living', data=df, ax=axes[1])
#sns.despine(bottom=True, left=True)
#axes[0].set(xlabel='Price in millions [USD]', ylabel='', title='Price Distribuition')
#axes[1].set(xlabel='Price', ylabel='Sqft Living', title='Price vs Sqft Living')
#axes[1].yaxis.set_label_position("right")
#axes[1].yaxis.tick_right()



#sns.set(style="whitegrid", font_scale=1)

#f, axes = plt.subplots(1, 2,figsize=(15,5))
#sns.boxplot(x=df['bedrooms'],y=df['price'], ax=axes[0])
#sns.boxplot(x=df['floors'],y=df['price'], ax=axes[1])
#sns.despine(bottom=True, left=True)
#axes[0].set(xlabel='Bedrooms', ylabel='Price', title='Bedrooms vs Price Box Plot')
#axes[1].set(xlabel='Floors', ylabel='Price', title='Floors vs Price Box Plot')




#f, axes = plt.subplots(1, 2,figsize=(15,5))
#sns.boxplot(x=df['waterfront'],y=df['price'], ax=axes[0])
#sns.boxplot(x=df['view'],y=df['price'], ax=axes[1])
#sns.despine(left=True, bottom=True)
#axes[0].set(xlabel='Waterfront', ylabel='Price', title='Waterfront vs Price Box Plot')
#axes[1].set(xlabel='View', ylabel='Price', title='View vs Price Box Plot')

#f, axe = plt.subplots(1, 1,figsize=(15,5))
#sns.boxplot(x=df['grade'],y=df['price'], ax=axe)
#sns.despine(left=True, bottom=True)
#axe.set(xlabel='Grade', ylabel='Price', title='Grade vs Price Box Plot')


df.isnull()


df = df.fillna(0)


#Export
df.to_csv('cleaned_kc.csv',index=False)
df







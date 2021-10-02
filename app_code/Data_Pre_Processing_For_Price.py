import pandas as pd
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
#get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
import scipy.stats as stats
pd.set_option('float_format', '{:f}'.format)

df = pd.read_csv(r'C:\Users\singa\Desktop\New folder\new\House-Price-Analysis\app_code\kc_house_data.csv') 

df.tail()

df.info()

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


# In[13]:


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


df.columns


df = df.drop(['sqft_living','sqft_lot','sqft_above','sqft_basement','zipcode'],axis=1)


df.dtypes


#correcting data types
##discrete vars as int

df.price = df.price.astype('int64')
df.bathrooms = df.bathrooms.astype('int64')
df.floors = df.floors.astype('int64')

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


#Export

df.to_csv('kc_cleaned_for_pricePredict.csv',index=False)
df




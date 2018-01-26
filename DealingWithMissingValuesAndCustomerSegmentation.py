
# coding: utf-8

# In[1]:

import pandas as pd
import numpy as np
import scipy
import matplotlib.pyplot as plt


# In[143]:

data = pd.read_csv('ByUsersDates_new.csv')


# In[144]:

pd.set_option('display.max_columns', None)
data

data['UserLoyalty'].value_counts()plt.bar(list(data['UserLoyalty'].value_counts().index), list(data['UserLoyalty'].value_counts()))
plt.show()
# In[145]:

#filling the unknown age groups with proportionate amounts of known age groups

props_list = [0.02, 0.16, 0.48, 0.22, 0.08, 0.03, 0.01]
new_numbers_list = []
age_groups_list = ['<18', '18-24', '25-34', '35-44', '45-54', '55-64', '65+']

for i in props_list:
    new_numbers_list.append(int(i*4962))
    
new_numbers_list[6] = new_numbers_list[6] + (4962-np.sum(new_numbers_list)) 
    

def isNaN(num):
    return num != num

x = 0
y = data['AgeGroup'].copy()


for i in range(0,len(new_numbers_list)):
    for j in range(0,len(y)-1):
        if x==new_numbers_list[i]:
            break
        else:
            if isNaN(y[j]) and x <= new_numbers_list[i]:
                y[j] = age_groups_list[i]
                x+=1
    x=0
    
data['AgeGroup'] = y


# ## Function to do one-hot-encoding

# In[2]:

def do_one_hot_encoding(df_name, df_column_name, suffix=''):
    x = pd.get_dummies(df_name[df_column_name])
    df_name = df_name.join(x, lsuffix=suffix)
    df_name = df_name.drop(df_column_name, axis=1) 
    return df_name


# In[147]:

data['AgeGroupLabels'] = label_encoding_func(data, 'AgeGroup')
data = do_one_hot_encoding(data, 'AgeGroup')


# In[148]:

data = do_one_hot_encoding(data, 'Gender')

YData1Year = data['Churn1Year']
YDataHalfYear = data['ChurnHalfYear']
# ## Label Encoding

# In[3]:

from sklearn import preprocessing

def label_encoding_func(df_name, df_col_name):
    le = preprocessing.LabelEncoder()
    le.fit(df_name[df_col_name])
    return le.transform(df_name[df_col_name])


# In[150]:

data = do_one_hot_encoding(data, 'MonthOfLastTransaction', '_last_month_trans')


# In[61]:

data


# In[4]:

from datetime import datetime, date

Y=2000
seasons = [('winter', (date(Y,  1,  1),  date(Y,  3, 20))),
           ('spring', (date(Y,  3, 21),  date(Y,  6, 20))),
           ('summer', (date(Y,  6, 21),  date(Y,  9, 22))),
           ('autumn', (date(Y,  9, 23),  date(Y, 12, 20))),
           ('winter', (date(Y, 12, 21),  date(Y, 12, 31)))]

def get_season(i):
    datetime_object = datetime.strptime(i, '%d.%m.%Y %H:%M')
    date_object = datetime_object.date()
    #Y = get_year(i)
    date_object = date_object.replace(year=Y)
    return next(season for season, (start, end) in seasons if start <= date_object <= end)

def get_year(i):
    datetime_object = datetime.strptime(i, '%d.%m.%Y %H:%M')
    return datetime_object.year


# In[152]:

data['FirstTransactionSeason'] = data['FirstTransactionDate(1)'].apply(get_season)


# In[153]:

data['LastTransactionSeason'] = data['LastTransactionDate(L)'].apply(get_season)


# In[154]:

data['FirstTransactionSeason'] = label_encoding_func(data, 'FirstTransactionSeason')


# In[155]:

data['LastTransactionSeason'] = label_encoding_func(data, 'LastTransactionSeason')


# In[156]:

data['FirstTransactionYear'] = data['FirstTransactionDate(1)'].apply(get_year)
data['LastTransactionYear'] = data['LastTransactionDate(L)'].apply(get_year)


# In[157]:

data = do_one_hot_encoding(data, 'FirstTransactionYear')
data = do_one_hot_encoding(data, 'LastTransactionYear', '_lt_trans_year')


# ## Finalize the dataset (Remove Useless Variables)

# In[158]:

data_col_names = data.columns.values
data_col_names


# In[274]:

X_data_col_names = ['ID_user', 'Age', 'CountTransactions', 'DaysBetweenFL(L-1)',
       'DaysBetweenLastTrAndToday',
       'AgeGroupOrder', 'StDevPrice', 'CountOrders',
       'TransactionsPerOrder', 'SpentPerOrder', 'SpentPerTransaction',
       'MoneySpentTotal', 'FirstTransactionDate(1)',
       'LastTransactionDate(L)', 'DistinctCountProducts',
       'DistinctCountProductCategories', 'MostFrequentCategory',
       'CountTransInMostFreqCat', 'MostSpentCategory',
       'SpentInMostSpentCat', 'DaysFrom1TrTo1YearChurn',
       'DaysFrom1TrTo05YearChurn', 'AgeGroupLabels', '18-24', '25-34',
       '35-44', '45-54', '55-64', '65+', '<18', 'female', 'male',
       'unknown', 'Apr', 'Aug', 'Dec', 'Feb', 'Jan', 'Jul', 'Jun', 'Mar',
       'May', 'Nov', 'Oct', 'Sep', 'FirstTransactionSeason',
       'LastTransactionSeason', '2010_lt_trans_year', '2011_lt_trans_year',
       '2012_lt_trans_year', '2013_lt_trans_year', '2014_lt_trans_year',
       '2015_lt_trans_year', '2010', '2011', '2012', '2013', '2014', '2015']

X_data = data[X_data_col_names]

X_data = X_data.fillna(np.mean(X_data))


# In[275]:

X_data.to_csv('CleanedData1yearChurn.csv', index=False)


# In[276]:

just_checking = pd.read_csv('CleanedData1yearChurn.csv')
just_checking


# # Models Start From Here

# In[286]:

X = pd.read_csv('CleanedData1yearChurn.csv', usecols=range(0,59))

X['SpentPerTransaction2'] = X['SpentPerTransaction']**2
X['Less18'] = X['<18']
X = X.drop('<18', axis=1)


# In[287]:

X


# ## Customer Segmentation using K-means

# In[279]:

from sklearn.cluster import KMeans


# In[289]:

X_data = X[['ID_user','CountTransactions', 'DaysBetweenFL(L-1)', 
            'DistinctCountProductCategories', 'MostSpentCategory', 'AgeGroupLabels']]


# In[291]:

cluster = KMeans(n_clusters=4)
X_data['cluster'] = cluster.fit_predict(X_data.ix[:,1:])
X_data.cluster.value_counts()


# In[292]:

X_clustered_data = X_data[['ID_user', 'cluster']]


# In[295]:

X = X.merge(X_clustered_data, on='ID_user')


# In[297]:

X.to_csv("new_clustered_data.csv", index=False, sep=",")


# In[ ]:

pd.DataFrame(X)

#feature importance for random forest classifier
import matplotlib.pyplot as plt

importances = rfm1.feature_importances_
std = np.std([tree.feature_importances_ for tree in rfm1.estimators_],
             axis=0)
indices = np.argsort(importances)[::-1]

# Print the feature ranking
print("Feature ranking:")

for f in range(train_x.shape[1]):
    print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))

# Plot the feature importances of the forest
plt.figure()
plt.title("Feature importances")
plt.bar(range(train_x.shape[1]), importances[indices],
       color="r", yerr=std[indices], align="center")
plt.xticks(range(train_x.shape[1]), indices)
plt.xlim([-1, train_x.shape[1]])

plt.show()
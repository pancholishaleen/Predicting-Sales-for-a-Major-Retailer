
# coding: utf-8

# In[2]:


# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# In[84]:


# Importing the dataset
dataset = pd.read_csv('C:\ML SimpliLearn\Project\Train_Kaggle.csv')
iv = dataset.iloc[:, 0: 3].values
dv = dataset.iloc[:, 3:4].values
print (iv)
print(dv)


# In[85]:


from sklearn.preprocessing import Imputer
missingValueImputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis=0)
missingValueImputer = missingValueImputer.fit(dv)
dv = missingValueImputer.fit_transform(dv)
print(dv)


# In[83]:


iv


# In[86]:


from sklearn.preprocessing import LabelEncoder
x_labelencoder = LabelEncoder()
iv[:, 2]=x_labelencoder.fit_transform(iv[:, 2])
print(iv)


# In[88]:


# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
iv_train, iv_test, dv_train, dv_test = train_test_split(iv, dv, test_size = 0.2, random_state = 0)
iv_train


# In[134]:


dataset = pd.read_csv('C:\ML SimpliLearn\Project\Test_Kaggle.csv')
x = dataset.iloc[:, 0: 3].values
y = dataset.iloc[:, 3:4].values
print (x)
print(y)


# In[137]:



x2_labelencoder = LabelEncoder()
x[:, 2]=x2_labelencoder.fit_transform(x[:, 2])
print(x)


# In[146]:


# Fitting Random Forest Classification to the Training set
from sklearn.svm import SVR
svr_regressor = SVR(kernel = 'linear', C=100000)
svr_regressor.fit(iv_train, dv_train)


# In[135]:


# Predicting the Test set results
y_pred = svr_regressor.predict(iv_test)
y_pred



# In[147]:


y = svr_regressor.predict(x)
prediction = pd.DataFrame(y, columns=['Sales(In ThousandDollars)']).to_csv('C:\ML SimpliLearn\Project\Shaleen_SVR.csv')


# In[109]:


from sklearn.metrics import mean_squared_error
from math import sqrt
rms = sqrt(mean_squared_error(iv_test, y_pred))
rms


# In[133]:


score = svr_regressor.score(iv_test, dv_test)
score


# In[139]:


from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 1000, random_state=0)
regressor.fit(iv_train, dv_train)


# In[141]:


score = regressor.score(iv_test, dv_test)
score


# In[142]:


y = regressor.predict(x)
y


# In[145]:


y_pred = regressor.predict(iv_test)
#y_pred.to_csv('C:\ML SimpliLearn\Project\shaleen_RF.csv')
prediction = pd.DataFrame(y_pred, columns=['Sales(In ThousandDollars)']).to_csv('C:\ML SimpliLearn\Project\Test_Kaggle.csv')


# In[104]:


rms = sqrt(mean_squared_error(dv_test, y_pred))
rms


# In[24]:





# In[25]:





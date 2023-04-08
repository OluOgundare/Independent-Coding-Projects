#!/usr/bin/env python
# coding: utf-8

# In[5]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
import zipfile
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# # Source
# 
# https://www.kaggle.com/datasets/parisrohan/credit-score-classification?resource=download

# In[6]:


df = pd.read_csv("train.csv.zip")
df["Monthly_Inhand_Salary"] = df["Monthly_Inhand_Salary"].fillna(df["Monthly_Inhand_Salary"].mean())


# In[11]:


df["Credit_Score"].unique()
df["Credit_Score_Num"] = 1
for x in range (0, len(df)):
    if df["Credit_Score"][x] == "Good":
        df["Credit_Score_Num"][x] = 1
    if df["Credit_Score"][x] == "Standard":
        df["Credit_Score_Num"][x] = 0
    if df["Credit_Score"][x] == "Poor":
        df["Credit_Score_Num"][x] = -1
df


# ### KNeighbors Model

# In[12]:


X = df[["Monthly_Inhand_Salary"]]
Y = df["Credit_Score"]

X_train,X_test,Y_train,Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 42)

model = KNeighborsClassifier(5)

model.fit(X_train,Y_train)

predictions_train = model.predict(X_train)
predictions_test = model.predict(X_test)

print("Training Accuracy:", accuracy_score(Y_train, predictions_train))
print("Testing Accuracy:", accuracy_score(Y_test, predictions_test))


# In[13]:


for x in range(0, len(df["Annual_Income"])):
    df["Annual_Income"][x] = df["Annual_Income"][x].replace("_", "")
    df["Num_of_Loan"][x] = df["Num_of_Loan"][x].replace("_", "")
df.head(5)


# ### Decision Tree Model

# In[ ]:


features = pd.DataFrame()
features["1"] = df[["Annual_Income"]]
features["2"] = df[['Num_of_Loan']]
features["3"] = df[['Num_Bank_Accounts']]
X = features
Y = df["Credit_Score"]

model = DecisionTreeClassifier(max_depth = 7)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.40, random_state = 30)

model.fit(X_train, Y_train)

predictions_train = model.predict(X_train)
predictions_test = model.predict(X_test)

print("Training Accuracy:", accuracy_score(Y_train, predictions_train))
print("Testing Accuracy:", accuracy_score(Y_test, predictions_test))


# In[ ]:





# In[ ]:





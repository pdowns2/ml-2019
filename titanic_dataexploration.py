#!/usr/bin/env python
# coding: utf-8

# In[13]:


import pandas as pd  
import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns  # pretty plotting, including heat map


# In[20]:


titanic_train=pd.read_csv('train.csv')
titanic_train.set_index("PassengerId", inplace=True)  #read in data and set passenger ID as the index


# In[34]:


titanic_train.head(10)


# In[114]:


titanic_train.describe()


# In[134]:


grouped_survival = titanic_train.groupby('Survived') #were children in fact let off first
round(grouped_survival.describe(),1)


# In[142]:


round(titanic_train.groupby('Survived')['Age'].mean(),1) #were children in fact let off first


# In[143]:


grouped_class = titanic_train.groupby('Pclass') #were children in fact let off first
round(grouped_class.describe(),1)


# In[144]:


a = titanic_train.groupby('Pclass')[['Survived']].count()
b = titanic_train.groupby('Pclass')[['Survived']].sum()
c = round((b/a)*100,1)
c.columns = ['Percent Survived']
c #what percent of each class survived


# In[145]:


titanic_train.plot(kind='scatter', x='Age', y='Pclass')
plt.show() # what is the age distribution across class? 


# In[173]:


pd.pivot_table(titanic_train,index=["Embarked"], columns= ["Pclass"], values=['Survived'], aggfunc = np.sum) 


# In[183]:


pt = pd.pivot_table(titanic_train,index=["Embarked"], columns= ["Pclass"], values=['Survived'], aggfunc = np.mean)
pt# percent that survived based on class and boarding location


# In[284]:


vals = pt.values
plt.plot([1, 2, 3], vals[0], label='Cherbourg')
plt.plot([1, 2, 3], vals[1], label='Queenstown')
plt.plot([1, 2, 3], vals[2], label='Southampton')
plt.legend(framealpha=1, frameon=True)
plt.xlabel('Class')
plt.ylabel('Survival Rate')
plt.xticks(range(1,4))
plt.title('Survival By Class and Departure City')
plt.savefig('Survival By Class and Departure City.png')
plt.show()


# 

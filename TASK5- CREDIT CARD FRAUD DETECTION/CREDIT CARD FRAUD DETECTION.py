#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, recall_score , classification_report


# In[2]:


cc_data= pd.read_csv(r"C:\Users\samru\OneDrive\Desktop\codsoft\creditcard.csv")


# In[3]:


cc_data.head()


# In[4]:


cc_data.info()


# In[5]:


cc_data.isnull().sum()


# In[6]:


cc_data['Class'].value_counts()


# In[7]:


legit=cc_data[cc_data.Class==0]
fraud=cc_data[cc_data.Class==1]


# In[8]:


print(legit.shape)
print(fraud.shape)


# In[9]:


legit.Amount.describe()


# In[10]:


fraud.Amount.describe()


# In[11]:


fraud.Amount.describe()


# In[12]:


legit_sample=legit.sample(n=261)


# In[13]:


new_data=pd.concat([legit_sample, fraud],axis=0)


# In[14]:


new_data.tail()


# In[15]:


new_data['Class'].value_counts()


# In[16]:


new_data.groupby('Class').mean()


# In[17]:


X=new_data.drop(columns='Class', axis=1)
Y=new_data['Class']


# In[18]:


print(X)


# In[19]:


print(Y)


# In[20]:


X_train,X_test,Y_train,Y_test=train_test_split (X,Y, test_size=0.2, stratify=Y,random_state=2)


# In[21]:


print(X.shape, X_train.shape,X_test.shape)


# In[22]:


model=LogisticRegression()


# In[36]:


model.fit(X_train, Y_train)


# In[27]:


X_train_prediction=model.predict(X_train)
training_data_accuracy=accuracy_score(X_train_prediction,Y_train)


# In[28]:


print('Accuracy on training data :{:.2f}%'.format(training_data_accuracy*100))


# In[29]:


X_test_prediction=model.predict(X_test)
test_data_accuracy=accuracy_score(X_test_prediction,Y_test)


# In[30]:


print('Accuracy on test data :{:.2f}%'.format(test_data_accuracy*100))


# In[31]:


print (classification_report(Y_test , model.predict(X_test)))


# In[ ]:





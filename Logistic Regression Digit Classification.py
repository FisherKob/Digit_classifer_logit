#!/usr/bin/env python
# coding: utf-8

# In this module, I will implement Logistic Regression using scikit-learn pre-loaded dataset. This is just a practice implementation. To work with large scale classificaion dataset that represents the real world ML tasks, please refer to the MNIST Handwritten digit database. There are several variations, but here is one of the links:http://yann.lecun.com/exdb/mnist/
# 

# In[1]:


# Each data point is 8*8 image


# In[49]:


from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
import numpy as np
import seaborn as sns
from sklearn import metrics
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[13]:


digits=load_digits()


# In[14]:


digits.data


# In[15]:


digits.data.shape


# In[16]:


digits.target.shape


# Showing the Images and Lables

# In[20]:


plt.figure(figsize=(20,4))
for index, (image,label) in enumerate(zip(digits.data[0:6],digits.target[0:6])):
    plt.subplot(1,6,index+1)
    plt.imshow(np.reshape(image, (8,8)),cmap=plt.cm.gray)
    plt.title('Training %i\n'%label,fontsize=10)
    


# Splitting the data into training and testing set

# In[21]:


x_train,x_test,y_train,y_test=train_test_split(digits.data,digits.target,test_size=0.2)


# In[22]:


x_train.shape


# In[23]:


y_train.shape


# Now, lets import logistic regression

# In[25]:


from sklearn.linear_model import LogisticRegression


# In[29]:


logistic


# In[26]:


logistic=LogisticRegression()


# In[28]:


logistic.fit(x_train,y_train)


# In[41]:


x_test[0].RESHAPE(0)


# In[36]:


x_test[0:10].shape


# In[43]:


logistic.predict(x_test[0].reshape(1,-1))


# In[32]:


logistic.predict(x_test[0:10])


# In[45]:


#accessingh the model performance
score=logistic.score(x_test,y_test)
score


# In[46]:


y_pred=logistic.predict(x_test)


# In[53]:


#Confusing matrix 
confusion_matrix(y_test, y_pred)


# In[58]:


plt.figure(figsize=(10,5))
sns.heatmap(confusion_matrix(y_test, y_pred),cmap='Blues_r')


# In[ ]:





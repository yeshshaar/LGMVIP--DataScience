#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
get_ipython().run_line_magic('matplotlib', 'inline')


# In[10]:


columns = ['Sepal length', 'Sepal width', 'Petal length', 'Petal width', 'Class_labels'] 
# Load the data
df = pd.read_csv("http://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data",names=columns)
df.head()


# In[11]:


df.describe()


# In[12]:


sns.pairplot(df, hue='Class_labels')


# In[13]:


data = df.values
X = data[:,0:4]
Y = data[:,4]


# In[14]:


Y_Data = np.array([np.average(X[:, i][Y==j].astype('float32')) for i in range (X.shape[1])
 for j in (np.unique(Y))])
Y_Data_reshaped = Y_Data.reshape(4, 3)
Y_Data_reshaped = np.swapaxes(Y_Data_reshaped, 0, 1)
X_axis = np.arange(len(columns)-1)
width = 0.25


# In[15]:


plt.bar(X_axis, Y_Data_reshaped[0], width, label = 'Setosa')
plt.bar(X_axis+width, Y_Data_reshaped[1], width, label = 'Versicolour')
plt.bar(X_axis+width*2, Y_Data_reshaped[2], width, label = 'Virginica')
plt.xticks(X_axis, columns[:4])
plt.xlabel("Features")
plt.ylabel("Value in cm.")
plt.legend(bbox_to_anchor=(1.3,1))
plt.show()


# In[16]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)


# In[17]:


from sklearn.svm import SVC
svn = SVC()
svn.fit(X_train, y_train)


# In[18]:


predictions = svn.predict(X_test)


# In[19]:


from sklearn.metrics import accuracy_score
accuracy_score(y_test, predictions)


# In[20]:


from sklearn.metrics import classification_report
print(classification_report(y_test, predictions))


# In[25]:


X_new = np.array([[4, 3, 1, 0.2], [  4.9, 2.3, 3.9, 1.1 ], [  5.3, 2.5, 4.7, 1.9 ]])


# In[26]:


prediction = svn.predict(X_new)
print("Prediction of Species: {}".format(prediction))


# In[27]:


import pickle
with open('SVM.pickle', 'wb') as f:
    pickle.dump(svn, f)


# In[28]:


with open('SVM.pickle', 'rb') as f:
    model = pickle.load(f)
model.predict(X_new)


# In[ ]:





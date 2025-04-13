#!/usr/bin/env python
# coding: utf-8

# In[2]:


#importing library
import numpy as np
import pandas as pd 


# In[3]:


#read dataset
data = pd.read_csv('C:\\Users\\kanga\\OneDrive\\Desktop\\Minor Project\\Dataset\\Crop_recommendation.csv')


# In[4]:


#data analysis
data.head(5)


# In[5]:


data.tail(5)


# In[6]:


data.shape


# In[7]:


data.info()


# In[8]:


print(data.corr())


# In[9]:


data.describe()


# In[10]:


data.isnull().sum()


# In[11]:


data.nunique()


# In[12]:


print(data['label'].nunique())
print(data['label'].unique())


# In[13]:


#label encoding
data['label']=data['label'].replace({'rice':0,'maize':1,'jute':2,'cotton':3,'coconut':4,'papaya':5,'orange':6,'apple':7,'muskmelon':8,'watermelon':9,'grapes':10,'mango':11,'banana':12,'pomegranate':13,'lentil':14,'blackgram':15,'mungbean':16,'mothbeans':17,'pigeonpeas':18,'kidneybeans':19,'chickpea':20,'coffee':21})
from sklearn.preprocessing import LabelEncoder
ls=LabelEncoder()


# In[14]:


print(data['label'].nunique())
print(data['label'].unique())
print(data['label'].value_counts())


# In[15]:


x=data.iloc[:,0:7].values
y=data.iloc[:,7:].values
x[:2],y[148:]


# In[16]:


#feature scaling
from sklearn.preprocessing import MinMaxScaler
mms=MinMaxScaler()
mmsx=mms.fit_transform(x)
print(mmsx[:5])
print(pd.DataFrame(mmsx).describe())


# In[17]:


#labelbinarizer
from keras.utils import np_utils
npy=np_utils.to_categorical(y)
print(npy[:5])


# In[18]:


#train_test_split
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(mmsx, npy, test_size = 0.4,random_state=42)


# In[19]:


print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)


# In[20]:


#importing libraries
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential 
from keras.layers import Dense,Activation
from keras.optimizers import Adam
from keras.metrics import categorical_crossentropy


# In[21]:


#building the model
model=Sequential()
model.add(Dense(1250,input_dim=7,activation='relu'))
#hidden layer
model.add(Dense(1300,activation='sigmoid'))
#Output layer 
model.add(Dense(22,activation='softmax'))


# In[22]:


model.compile(optimizer='Adam',loss='categorical_crossentropy',metrics=['accuracy'])


# In[23]:


model.summary()


# In[24]:


#model.fit(x_train,y_train,epochs=100)

model.fit(x_train, y_train, epochs=10, validation_data=(x_test, y_test))


# In[25]:


pred_y=model.predict(x_test)
pred_y=np.argmax(pred_y,axis=1)
print(pred_y)
y_test=np.argmax(y_test,axis=1)

print(y_test)


# In[26]:


from sklearn.metrics import classification_report,confusion_matrix,accuracy_score

print(classification_report(pred_y,y_test))
print(confusion_matrix(pred_y,y_test))
print(accuracy_score(pred_y,y_test))


# In[27]:


x_test[0:1]


# In[ ]:





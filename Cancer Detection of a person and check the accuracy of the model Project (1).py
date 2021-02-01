#!/usr/bin/env python
# coding: utf-8

# In[102]:


import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


# In[103]:


#load the data
df=pd.read_csv('data (2) (2).csv')
df.head(10)


# In[104]:


df.shape


# In[105]:


#count the number of all empty values in each columns
df.isna().sum()


# In[106]:


#drop the column with all missing values
df=df.dropna(axis=1)


# In[107]:


#get the new count of the number of rows and columns
df.shape


# In[108]:


#get the count of the number of malognant(m) and benign(b)
df['diagnosis'].values


# In[109]:


df['diagnosis'].value_counts()


# In[110]:


#visualize he count of the diagnosis data
sns.countplot(df['diagnosis'],label='count')


# In[111]:


#look at the data typesbto see which columns need to be encoded
df.dtypes


# In[112]:


#encode the object 
from sklearn.preprocessing import LabelEncoder
lb=LabelEncoder()
df.iloc[:,1]=lb.fit_transform(df.iloc[:,1].values)# : indicates the all rows of the column and 1 is the second index i.e, diagnosis
df.iloc[:,1]


# In[113]:


sns.pairplot(df.iloc[:,1:7],hue='diagnosis') #for multiverate analysis


# In[116]:


df.head(5) #the new dataset after the encoded where diagnsis have became object to encoded value!!


# In[117]:


#get the corelation of the column
df.iloc[:,1:12].corr() #each rows mean on the column, whether it positive mean or negative


# In[118]:


#visualize the corelation
sns.heatmap(df.iloc[:,1:12].corr()) #we can see the diagnosis mean on the diagnosis column is 1.0 as per the above mean_dataset


# In[119]:


#more visualize
plt.figure(figsize=(10,10))
sns.heatmap(df.iloc[:,1:12].corr(),annot=True)


# In[120]:


#the exact percentage of the given dataset of finding M or B
plt.figure(figsize=(10,10))
sns.heatmap(df.iloc[:,1:12].corr(),annot=True,fmt='.0%')


# In[121]:


#spliting the dataset into x and y where x is independent and y is dependent
x=df.iloc[:,2:31].values #after diagnosis column
y= df.iloc[:,1].values
x


# In[122]:


y


# In[123]:


#split the data into training and testing
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25, random_state=0)#25% data is in test and the else part train!


# In[124]:


#scale the data i.e, find the magnitude of the data which means the portion of the data lies whether in 0-100 range or 0-1 
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
x_train=sc.fit_transform(x_train)
x_test=sc.fit_transform(x_test)
print(y_test)
print(x_test)


# In[125]:


#create a function for model
def ML_M(x_train,y_train):
    log=LogisticRegression(random_state=0)
    log.fit(x_train,y_train)
    tree=DecisionTreeClassifier(criterion='entropy',random_state=0)
    tree.fit(x_train,y_train)
    rf=RandomForestClassifier(n_estimators=10,criterion='entropy',random_state=0)
    rf.fit(x_train,y_train)
    print('[0]accuracy of the model for logistic regression =',log.score(x_train,y_train))
    print('[1accuracy of the model for decision tree classifier =',tree.score(x_train,y_train))
    print('[2]accuracy of the model for random fore classifier =',rf.score(x_train,y_train))
    return log,tree,rf
model=ML_M(x_train,y_train)
        
         


# In[126]:


y1_pred=model[0].predict(x_test) #testing part of model 1
print(y1_pred)


# In[127]:


y2_pred=model[1].predict(x_test) #testing part of model2
print(y2_pred)


# In[128]:


y3_pred=model[2].predict(x_test) #testing part of model 3
print(y3_pred)


# In[129]:


count_missclassified=(y_test!=y1_pred).sum() #missclassifies value between x_test and y_test logisticregressionclassifier
print('missclassified samples:{}'.format(count_missclassified))


# In[130]:


count_missclassified=(y_test!=y2_pred).sum() #missclassified value between x_test and y_test decisiontreeclassifier
print('missclassified samples:{}'.format(count_missclassified))


# In[131]:


count_missclassified=(y_test!=y3_pred).sum() #missclassified value between x_test and y_test randomforestclassifier
print('missclassified samples:{}'.format(count_missclassified))


# In[132]:


print(y_test)#print the y_test after standarddeviation


# In[133]:


#making the confusion matrix of model 0 to find true +ve ,false +ve, true -ve, false -ve values
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y1_pred)
cm


# In[134]:


#making the confusion matrix of model 1 to find true +ve ,false +ve, true -ve, false -ve values
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y2_pred)
cm


# In[135]:


#making the confusion matrix of model 2 to find true +ve ,false +ve, true -ve, false -ve values
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y3_pred)
cm


# In[136]:


#print the accuracy between y_test and predicted tested values
from sklearn.metrics import accuracy_score
accuracy1=accuracy_score(y_test,y1_pred)#for model 1
accuracy1


# In[137]:


#print the accuracy between y_test and predicted tested values
from sklearn.metrics import accuracy_score
accuracy2=accuracy_score(y_test,y2_pred)#for model 2
accuracy2


# In[138]:


#print the accuracy between y_test and predicted tested values
from sklearn.metrics import accuracy_score
accuracy3=accuracy_score(y_test,y3_pred)#for model 3
accuracy3


# In[139]:


#print the exact prediction of logisticregression model 
print(y1_pred)
print()
print(y_test)


# In[140]:


#print the exact prediction of decisiontreeclassifier model 
print(y2_pred)
print()
print(y_test)


# In[141]:


#print the exact prediction of randomforestclassifier model 
print(y3_pred)
print()
print(y_test)


# from the above 3 analyzing its going to be justified the missclassified data of the predicted tested model.  

# Project on Cancer Detection of a person.

# In[ ]:





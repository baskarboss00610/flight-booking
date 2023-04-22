#!/usr/bin/env python
# coding: utf-8

# In[5]:



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report,confusion_matrix
import warnings
import pickle
from scipy import stats
warnings.filterwarnings('ignore')
plt.style.use('fivethirtyeight')


# In[6]:


data=pd.read_csv("Data_Train.csv")
data.head()


# In[7]:


data.Date_of_Journey=data.Date_of_Journey.str.split('/')


# In[8]:


data.Date_of_Journey


# In[9]:


#Treating the data_column

data['Date']=data.Date_of_Journey.str[0]
data['Month']=data.Date_of_Journey.str[1]
data['Year']=data.Date_of_Journey.str[2]


# In[10]:


data.Total_Stops.unique()


# In[11]:


#Since the maximum number of is 4,there should be maximum 6 cities in any particular toute.we splite the data in route column

data.Route=data.Route.str.split('→')
data.Route


# In[12]:


data['City1']=data.Route.str[0]
data['City2']=data.Route.str[1]
data['City3']=data.Route.str[2]
data['City4']=data.Route.str[3]
data['City5']=data.Route.str[4]
data['City6']=data.Route.str[5]


# In[13]:


# In the similar manner,we split the Dep_time,and create separate for departure hours and minutes
data.Dep_Time=data.Dep_Time.str.split(":")


# In[14]:


data['Dep_Time_Hour']=data.Dep_Time.str[0]
data['Dep_Time_Mins']=data.Dep_Time.str[1]


# In[15]:


data.Arrival_Time=data.Arrival_Time.str.split(' ')


# In[16]:


data['Arrival_date']=data.Arrival_Time.str[1]
data['Time_of_Arrival']=data.Arrival_Time.str[0]


# In[17]:


data['Time_of_Arrival']=data.Time_of_Arrival.str.split(':')


# In[18]:


data['Arrival_Time_Hour']=data.Time_of_Arrival.str[0]
data['Arrival_Time_Mins']=data.Time_of_Arrival.str[1]


# In[19]:


# Next, we dividednthe 'Duration' column to 'Travel_hours' and 'Travel_mins' 
data.Duration=data.Duration.str.split(" ")


# In[20]:


data['Travel_Hours']=data.Duration.str[0]
data['Travel_Hours']=data['Travel_Hours'].str.split('h')
data['Travel_Hours']=data['Travel_Hours'].str[0]
data.Travel_Hours=data.Travel_Hours
data['Travel_Mins']=data.Duration.str[1]

data.Travel_Mins=data.Travel_Mins.str.split('m')
data.Travel_Mins=data.Travel_Mins.str[0]


# In[21]:


data.Total_Stops.replace('non_stop',0,inplace=True)
data.Total_Stops=data.Total_Stops.str.split(' ')
data.Total_Stops=data.Total_Stops.str[0]


# In[22]:


data.Total_Stops.replace('non_stop',0,inplace=True)
data.Total_Stops=data.Total_Stops.str.split(' ')
data.Total_Stops=data.Total_Stops.str[0]


# In[23]:


data.Additional_Info.unique()


# In[24]:


data.Additional_Info.replace('No Info','No info',inplace=True)


# In[25]:


data.isnull().sum()


# In[26]:


data.drop(['City4','City5','City6'],axis=1,inplace=True)


# In[27]:


data.drop(['Date_of_Journey','Route','Dep_Time','Arrival_Time','Duration'],axis=1,inplace=True)
data.drop(['Time_of_Arrival'], axis=1,inplace=True)


# In[28]:


#Checking Null Values
data.isnull().sum()


# In[29]:


data['City3'].fillna('None',inplace=True)


# In[30]:


data['Arrival_date'].fillna(data['Date'],inplace=True)


# In[31]:


data['Travel_Mins'].fillna(0,inplace=True)


# In[32]:


data.info()


# In[33]:


data.Date=data.Date.astype('int64')
data.Month=data.Month.astype('int64')
data.Year=data.Year.astype('int64')
data.Dep_Time_Hour=data.Dep_Time_Hour.astype('int64')
data.Dep_Time_Hour=data.Dep_Time_Hour.astype('int64')
data.Dep_Time_Mins=data.Dep_Time_Mins.astype('int64')
data.Arrival_date=data.Arrival_date.astype('int64')
data.Arrival_Time_Hour=data.Arrival_Time_Hour.astype('int64')
data.Arrival_Time_Mins=data.Arrival_Time_Mins.astype('int64')
data.Travel_Mins=data.Travel_Mins.astype('int64')


# In[34]:


data[data['Travel_Hours']=='5m']


# In[35]:


data.drop(index=6474,inplace=True,axis=0)


# In[36]:


data.Travel_Hours=data.Travel_Hours.astype('int64')


# In[37]:


categorical=['Airline','Source','Destination','Additional_Info','City1']
numerical=['Total_stops','Date','Month','Year','Dep_Time_Hour','Dep_Time_Mins','Arrival_date','Arrival_Time_Hour','Arrival_Time_Mins','Travel_Hours','Travel_Mins']


# In[38]:


from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()


# In[39]:


data.Airline=le.fit_transform(data.Airline)
data.Source=le.fit_transform(data.Source)
data.Destination=le.fit_transform(data.Destination)
data.Total_Stops=le.fit_transform(data.Total_Stops)
data.City1=le.fit_transform(data.City1)
data.City2=le.fit_transform(data.City2)
data.City3=le.fit_transform(data.City3)
data.Additional_Info=le.fit_transform(data.Additional_Info)
data.head()


# In[40]:


data.head()


# In[41]:


data=data[['Airline','Source','Destination','Date','Month','Year','Dep_Time_Hour','Dep_Time_Mins','Arrival_date','Arrival_Time_Hour','Arrival_Time_Mins','Additional_Info','City1','Price']]


# In[42]:


data.head()


# In[43]:


data.describe()


# In[44]:


import seaborn as sns
c=1
plt.figure(figsize=(20,45))

for i in categorical:
    plt.subplot(6,3,c)
    sns.countplot(data[i])
    plt.xticks(rotation=90)
    plt.tight_layout(pad=3.0)
    c=c+1

plt.show()


# In[45]:


plt.figure(figsize=(10,8))
sns.distplot(data.Price)


# In[46]:


sns.heatmap(data.corr(),annot=True)


# In[47]:


import seaborn as sns
sns.boxplot(data['Price'])


# In[48]:


y=data['Price']
x=data.drop(columns=['Price'],axis=1)


# In[49]:


from sklearn.preprocessing import StandardScaler
ss=StandardScaler()


# In[50]:


x_scaled=ss.fit_transform(x)


# In[51]:


x_scaled=pd.DataFrame(x_scaled,columns=x.columns)
x_scaled.head()


# In[52]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)


# In[53]:


x_train.head()


# In[54]:


from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor,AdaBoostRegressor
rfr=RandomForestRegressor()
gb=GradientBoostingRegressor()
ad=AdaBoostRegressor()


# In[55]:


from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error

for i in [rfr,gb,ad]:
    i.fit(x_train,y_train)
    y_pred=i.predict(x_test)
    test_score=r2_score(y_test,y_pred)
    train_score=r2_score(y_train,i.predict(x_train))
    if abs(train_score-test_score)<=0.2:
        print(i)

        print("R2 score is",r2_score(y_test,y_pred))
        print("R2 for train data",r2_score(y_train,i.predict(x_train)))
        print("Mean Absolute Error is",mean_absolute_error(y_pred,y_test))
        print("Mean Squre Error is",mean_squared_error(y_pred,y_test))
        print("Root Mean Squre Error is",(mean_squared_error(y_pred,y_test,squared=False)))
        


# In[56]:


from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error

knn=KNeighborsRegressor()
svr=SVR()
dt=DecisionTreeRegressor()

for i in [knn,svr,dt]:
    i.fit(x_train,y_train)
    y_pred=i.predict(x_test)
    test_score=r2_score(y_test,y_pred)
    train_score=r2_score(y_train,i.predict(x_train))
    if abs(train_score-test_score)<=0.1:
        print(i)

        print("R2 score is",r2_score(y_test,y_pred))
        print("R2 for train data",r2_score(y_train,i.predict(x_train)))
        print("Mean Absolute Error is",mean_absolute_error(y_test,y_pred))
        print("Mean Squre Error is",mean_squared_error(y_test,y_pred))
        print("Root Mean Squre Error is",(mean_squared_error(y_test,y_pred,squared=False)))
        


# In[57]:


from sklearn.model_selection import cross_val_score
for i in range(2,5):
    cv=cross_val_score(rfr,x,y,cv=i)
    print(rfr,cv.mean())


# In[58]:


from sklearn.model_selection import RandomizedSearchCV


# In[59]:


param_grid={'n_estimators':[10,30,50,70,100],'max_depth':[None,1,2,3],'max_features':['auto','sqrt']}
rfr=RandomForestRegressor()
rf_res=RandomizedSearchCV(estimator=rfr,param_distributions=param_grid,cv=3,verbose=2,n_jobs=-1)

rf_res.fit(x_train,y_train)


# In[60]:


gb=GradientBoostingRegressor()
gb_res=RandomizedSearchCV(estimator=gb,param_distributions=param_grid,cv=3,verbose=2,n_jobs=-1)

gb_res.fit(x_train,y_train)


# In[61]:


rfr=RandomForestRegressor(n_estimators=10,max_features='sqrt',max_depth=None)
rfr.fit(x_train,y_train)
y_train_pred=rfr.predict(x_train)
y_test_pred=rfr.predict(x_test)
print("train accuracy",r2_score(y_train_pred,y_train))
print("test accuracy",r2_score(y_test_pred,y_test))


# In[62]:


knn=KNeighborsRegressor(n_neighbors=2,algorithm='auto',metric_params=None,n_jobs=-1)
knn.fit(x_train,y_train)
y_train_pred=knn.predict(x_train)
y_test_pred=knn.predict(x_test)
print("train accuracy",r2_score(y_train_pred,y_train))
print("test accuracy",r2_score(y_test_pred,y_test))


# In[63]:


rfr=RandomForestRegressor(n_estimators=10,max_features='sqrt',max_depth=None)
rfr.fit(x_train,y_train)
y_train_pred=knn.predict(x_train)
y_test_pred=knn.predict(x_test)
print("train accuracy",r2_score(y_train_pred,y_train))
print("test accuracy",r2_score(y_test_pred,y_test))


# In[64]:


prices=y_test_pred


# In[65]:


price_list=pd.DataFrame({'Price':prices})
price_list


# In[66]:


import pickle
pickle.dump(rfr,open('model1.pkl','wb'))


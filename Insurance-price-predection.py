#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import index.jade.form. as response 
import matplotlib.pyplot as plt
from matplotlib.pyplot import rcParams
import seaborn as sns
import numpy as np
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder,OrdinalEncoder,StandardScaler,MinMaxScaler,PowerTransformer,FunctionTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection  import train_test_split
from sklearn.metrics import mean_squared_error
from scipy import stats
import warnings
warnings.filterwarnings('ignore')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


#read data

df = pd.read_csv('./insurance_data.csv')
df


# In[3]:


df.info()


# In[4]:


#check for missing values 
def missing_values(df):
    null_v = df.isnull().sum().sort_values(ascending=False)
    null_percent = (df.isnull().sum()/df.isnull().count()).sort_values(ascending=False)
    null_v = pd.concat([null_v, null_percent], axis=1, keys=['Missing_Number', 'Missing_Percent'])
    return null_v

missing_values(df)


# In[5]:


#check for duplicated rows

df.duplicated().sum()


# In[6]:


#showing duplicated rows
df[df.duplicated(keep=False)]


# In[7]:


#dropping the duplicated rows 
df=df.drop_duplicates(keep="first")


# In[8]:


df.duplicated().sum()


# In[9]:


df.describe()


# In[10]:


#check data description
df.describe().T.style.bar()


# In[11]:


#locating row that has the max price 
df.loc[df['charges'] == 63770.428010]


# In[12]:


#locating row that has the max BMI
df.loc[df['bmi'] == 53.130000]


# In[13]:


#locating row that show people who have max children of 5
df.loc[df['children'] == 5.000000]


# In[14]:


#check the correlation between charges and each of columns
df.corr()["charges"]


# In[15]:


#analysis sex vs smoker columns combinely 
sns.countplot(df["sex"],hue=df["smoker"],palette="Set1")


# In[16]:


#plotting data for visualization

catergoy_column=["sex","smoker","region","children"]

colors=["#00FFFF","#FFA597","#00CFFC","#ED00D9","#ADD8E6","#EFF999"]
textprops = {"fontsize":22}

plt.figure(figsize=(25,90))
i=1
for column in catergoy_column:
    plt.subplot(11,2,i)
    sns.countplot(data=df,x=column,palette="Spectral")
    plt.xticks(fontsize=25)
    plt.yticks(fontsize=25)
    plt.xlabel(column,fontsize=25)
    plt.ylabel("count",fontsize=25)
    i=i+1
    plt.subplot(11,2,i)
    df[column].value_counts().plot(kind="pie",autopct="%.2f%%",colors=colors,textprops=textprops,radius = 1.1)
    plt.xticks(fontsize=25)
    plt.yticks(fontsize=25)
    plt.xlabel(column,fontsize=25)
    plt.ylabel("count",fontsize=25)
    i=i+1

plt.show()


# In[17]:


#Check for distribution of charges
sns.distplot(df['charges'])


# In[18]:


q = df['charges'].quantile(0.99)


# In[19]:


#analysis age and bmi columns
plt.figure(figsize=(18,8))
plt.subplot(121)
sns.histplot(df["age"],color="#ED00D9",fill=True)
plt.title("Age")
plt.xticks(fontsize=25)
plt.yticks(fontsize=25)
plt.xlabel("Age",fontsize=25)
plt.ylabel("count",fontsize=25)
    

plt.subplot(122)
sns.histplot(df["bmi"],color="#00FFFF")
plt.title("BMI")
plt.xticks(fontsize=25)
plt.yticks(fontsize=25)
plt.xlabel("BMI",fontsize=25)
plt.ylabel("count",fontsize=25)
   

plt.show()


# In[20]:


#checking for outliers in age col

sns.boxplot(df["age"],palette="Spectral")
            
#NO outliers in AGE
            


# In[21]:


#outliers in bmi col...
sns.boxplot(df["bmi"],palette="Spectral")
           
           
#There are outliers in BMI

#Outliers increase the variability in your data, which decreases statistical power. 


# In[22]:


#Check for distribution of BMI
sns.distplot(df['bmi'])


# In[23]:


#REMOVING OUTLIERS TO TRAIN DATA



def outlier(data):

    mean=data.mean()
    std=data.std()
    mini=data.min()
    maxi=data.max()

    #let find the boundaries for outlier
    highest=data.mean() + 3*data.std()
    lowest=data.mean() - 3*data.std()

        #finally, let find the outlier
    outliers=df[(data>highest) | (data<lowest)]
        

    return outliers
#outliers detection and remove  
new=pd.DataFrame(df["bmi"],columns=["bmi"])
for col in new.columns:
    test=outlier(df[col])
    print("columns name :",col)
    print("numbers of outliers:",len(test))
    print("\n")
    print(test)
    print("<<<<<<<<<------------------------------------->>>>>>>>>")
    
#drop the outliers by thier index    
    df=df.drop(test.index,axis=0)                        


# In[24]:


# CHANGE BMI to a Catogory for data training

#function that will change  bmi to a category
def weightCondition(bmi):
  if bmi<18.5:
    return "Underweight"
  elif (bmi>= 18.5)&(bmi< 24.986):
    return "Normal"
  elif (bmi >= 25) & (bmi < 29.926):
    return "Overweight"
  else:
    return "Obese"
df["weight_Condition"]=[weightCondition(val) for val in df["bmi"] ]
df.head(5)


# In[25]:


#TRAINING DATA

#get the features and target col
Y=df.charges
X=df.drop(["charges"],axis=1)
#train test split  
x_train,x_test,y_train,y_test=train_test_split( X,Y,test_size=0.2,random_state=42)


# In[26]:


x_train.head()


# In[27]:


x_test.head()


# In[28]:


y_train.head()


# In[29]:


#Checking for types of Weight Condition
x_train["weight_Condition"].unique()


# In[30]:


#Machine learning algorithms cannot work with categorical data directly.

#Categorical data must be converted to numbers.

#pipe1 contain 2 encoder ,one hot encoder and ordinal encoder
#one hot encoder includes sex,smoker,region, weight_condition
#ordinal encode the weight_condition col because we arrange the order on this col
#pipe2 just scale all the columns 
pipe1=ColumnTransformer(transformers=[("OHE",OneHotEncoder(sparse=False,drop="first"),
                                       ["sex","smoker","region"]),
                                     ("ordinal",OrdinalEncoder(categories=[['Underweight','Normal','Overweight','Obese']]),
                                      ["weight_Condition"])]
                        ,remainder="passthrough")
pipe2=ColumnTransformer(transformers=[("scaling",StandardScaler(),[0,1,2,3,4,5,6,7,8])],
                        remainder="passthrough")

pipe=Pipeline([("pipe1",pipe1),("pipe2",pipe2)])
x_train=pd.DataFrame(pipe.fit_transform(x_train))
x_test=pd.DataFrame(pipe.transform(x_test))
x_train.head()


# In[31]:


# LOGIC TO TEST NEW DATA ROW
reg = LinearRegression().fit(x_train, y_train)

response_array = [response[0],response[1], response[2], response[3], response[4], response[5]]

response_df = pd.DataFrame([response_array], columns = ['age', 'sex', 'bmi', 'children', 'smoker', 'region'])
response_df["weight_Condition"]=[weightCondition(val) for val in response_df['bmi']]


encoded_response_df = pd.DataFrame(pipe.transform(response_df))
y_pred = reg.predict(encoded_response_df) 

y_pred

export y_pred 
        


# In[ ]:





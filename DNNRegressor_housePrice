# -*- coding: utf-8 -*-
#"""
#Spyder Editor

#This is a temporary script file.
#"""

from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from matplotlib import pyplot as plt
from sklearn.grid_search import GridSearchCV
import pandas as pd
import numpy as np
from sklearn.model_selection import  train_test_split
from sklearn.metrics import mean_squared_error
import tensorflow as tf

TrainingDataCost = []
TrainingDatapledged = []
trainlabels=[]
TestDataCost = []
TestDatapledged = []
Testlabels=[]
scores = []
columns_isnull = []
columns_isnull_butUsable=[]
testlabels=pd.DataFrame()

#Import Data    
train = open('C:/Users/mandar/Desktop/KaggleData/housing/train.csv')
df_t=pd.read_csv(train)  
Test = open('C:/Users/mandar/Desktop/KaggleData/housing/test.csv')
df_Test=pd.read_csv(Test)


for col in df_t.columns:
    pct=df_t[col].isnull().sum()/len(df_t)   
    if pct>0:
        columns_isnull.append(col)
               
   
           
df_train=df_t
df_test =df_Test          
#dropping the null value columns 
df_train.drop(columns_isnull, axis=1, inplace=True)
df_test.drop(columns_isnull, axis=1, inplace=True)    
#Slising from df only object data types columns: object_df
object_df = df_train.select_dtypes(include=['object']).copy()
for col in object_df.columns:
    df_train[col] = df_train[col].fillna("None")
    df_test[col] = df_train[col].fillna("None")            
for col in df_train.columns:
    if  df_train[col].dtype != 'object' and col!='SalePrice':
        df_train[col].fillna(df_train[col].mode()[0] , inplace=True)    
        df_test[col].fillna(df_test[col].mode()[0] , inplace=True)
#Create numerical features from categorical 
#Dropping the object columns:
object_columns = object_df.columns

df_train.drop(object_columns, axis=1, inplace=True)

dummie_test = pd.get_dummies(df_test[object_columns])
df_test.drop(object_columns, axis=1, inplace=True)

dummie_train = pd.get_dummies(object_df) 
df_t = pd.concat([df_train, dummie_train], axis=1)
df_Test = pd.concat([df_test, dummie_test], axis=1)
ExtraColumn_list = np.setdiff1d(df_Test.columns,df_t.columns)
df_Test.drop(ExtraColumn_list, axis=1, inplace=True)

col_train_bis = list(df_t.columns)
col_train_bis.remove('SalePrice')
col_train_bis.remove('Id')
col_train_bis.remove('1stFlrSF')
col_train_bis.remove('2ndFlrSF')
col_train_bis.remove('RoofMatl_Tar&Grv')
col_train_bis.remove('Exterior1st_Wd Sdng')
df_t.rename(index=str, columns={"MSZoning_C (all)": "MSZoning_C_(all)","Exterior2nd_Wd Shng": "Exterior2nd_Wd_Shng", "Exterior2nd_Brk Cmn": "Exterior2nd_Brk_Cmn","Exterior2nd_Wd Sdng": "Exterior2nd_Wd_Sdng"})
df_Test.rename(index=str, columns={"MSZoning_C (all)": "MSZoning_C_(all)","Exterior2nd_Wd Shng": "Exterior2nd_Wd_Shng", "Exterior2nd_Brk Cmn": "Exterior2nd_Brk_Cmn","Exterior2nd_Wd Sdng": "Exterior2nd_Wd_Sdng"})

FEATURES = col_train_bis
LABEL = "SalePrice"

# Columns for tensorflow
feature_cols = [tf.contrib.layers.real_valued_column(k) for k in FEATURES]

# Training set and Prediction set with the features to predict        
testlabels['Id']=df_Test['Id']
trainlabel=df_train.loc[0:,['SalePrice']]
    
traindat=df_t.loc[0:,col_train_bis]
testdat=df_Test.loc[0:,col_train_bis]
train_data, test_data,train_label,test_label  = train_test_split(traindat,trainlabel, test_size=0.1,random_state=0)


regressor = tf.contrib.learn.DNNRegressor(feature_columns=feature_cols, 
                                          activation_fn = tf.nn.relu, hidden_units=[200, 100, 50, 25, 12])

def input_fn(data_set,label_set, pred = False):    
    if pred == False:
        feature_ = {k: tf.constant(data_set[k].values) for k in FEATURES}
        labels_ = tf.constant(label_set)        
        return feature_,labels_  
        
    if pred == True:
        feature_ =  {k: tf.constant(data_set[k].values) for k in FEATURES} 
        return feature_ 

def evaluate():
    regressor.fit(input_fn=lambda: input_fn(train_data,train_label), steps=1200)
    ev = regressor.evaluate(input_fn=lambda: input_fn(test_data,test_label), steps=1)
    loss_score1 = ev["loss"]
    print (loss_score1)

def Pred():
    regressor.fit(input_fn=lambda: input_fn(traindat,trainlabel), steps=1300)
    predictions = regressor.predict(input_fn=lambda: input_fn(testdat,testdat,pred = True))        
    testlabels['SalePrice']=list(predictions)
    testlabels.to_csv('C:/Users/mandar/Desktop/KaggleData/housing/resulstDNNREGRESSION.csv')
Pred()

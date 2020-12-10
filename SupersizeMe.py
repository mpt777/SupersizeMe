# -*- coding: utf-8 -*-
"""
Created on Mon Dec  7 11:22:40 2020

@author: mptay
"""
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler, StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

from sklearn.impute import KNNImputer
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score, roc_curve, accuracy_score
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder, label_binarize
#from sklearn import preprocessing
#from sklearn.preprocessing import OneHotEncoder

from tabula import read_pdf
import PyPDF2

def CategoryFinder(x):
    df = []
    if "Burger" in x:
        df.append("Beef & Pork")
    elif "burger" in x:
        df.append("Beef & Pork")
    elif "Onion" in x:
        df.append("Beef & Pork")
    elif "with" in x:
        df.append("Beef & Pork")
    elif "Style" in x:
        df.append("Beef & Pork")
    elif "Fries" in x:
        df.append("Snacks & Sides")
    elif "Shake" in x:
        df.append("Smoothies & Shakes")
    elif "Coffee" in x:
        df.append("Coffee & Tea")
    else:
        df.append("Beverages")
    return(df)

def CategoryFinderRR(x):
    df = []
    if "burger" in x.lower():
        df.append("Beef & Pork")
    elif "caes" in x.lower():
        df.append("Salads")
    elif "burger" in x.lower():
        df.append("Beef & Pork")
    elif "patty" in x.lower():
        df.append("Beef & Pork")
    elif "ring" in x.lower():
        df.append("Snacks & Sides")
    elif "sticks" in x.lower():
        df.append("Snacks & Sides")
    elif "onion" in x.lower():
        df.append("Beef & Pork")
    elif "bites" in x.lower():
        df.append("Snacks & Sides")
    elif "style" in x.lower():
        df.append("Beef & Pork")
    elif "frie" in x.lower():
        df.append("Snacks & Sides")
    elif "shake" in x.lower():
        df.append("Smoothies & Shakes")
    elif "cookie" in x.lower():
        df.append("Smoothies & Shakes")
    elif "veggie" in x.lower():
        df.append("Beef & Pork")
    elif "robin" in x.lower():
        df.append("Beef & Pork")
    elif "burnin" in x.lower():
        df.append("Beef & Pork")
    elif "bacon" in x.lower():
        df.append("Beef & Pork")
    elif "wings" in x.lower():
        df.append("Chicken & Fish")
    elif "coffee" in x.lower():
        df.append("Coffee & Tea")
    elif "salad" in x.lower():
        df.append("Salads")
    elif "dress" in x.lower():
        df.append("Salads")
        
    elif "simple" in x.lower():
        df.append("Beef & Pork")
    elif "shroom" in x.lower():
        df.append("Beef & Pork")
    elif "bbq" in x.lower():
        df.append("Beef & Pork")
    elif "&" in x.lower():
        df.append("Beef & Pork")
    elif "cheese" in x.lower():
        df.append("Beef & Pork")
        
    elif "banzai" in x.lower():
        df.append("Beef & Pork")
        
    else:
        df.append("Beverages")
    return(df)

RedRobin = read_pdf("RedRobin.pdf",pages='all',multiple_tables = True, pandas_options={'header': None})

RedRobin_df = RedRobin[1].append( [RedRobin[13], RedRobin[12], RedRobin[8], RedRobin[6], RedRobin[2], RedRobin[24], RedRobin[23], RedRobin[14] ] )
RedRobin_df = RedRobin_df.reset_index()

RedRobin_df = RedRobin_df.drop('index',1)
RedRobin_df = RedRobin_df.drop_duplicates(subset=[0],keep='first')
RedRobin_df = RedRobin_df.drop([12,17],0)

RedRobin_df = RedRobin_df.reset_index()
RedRobin_df = RedRobin_df.drop('index',1)

RedRobin_df.iloc[64:71,1:13] = RedRobin_df.iloc[64:71,1:13].shift(1, axis = 1)
RedRobin_df.iloc[71:73,:] = RedRobin_df.iloc[71:73,:].shift(2, axis = 1)

RedRobin_df.iloc[16:37,6] = RedRobin_df.iloc[16:37,5]
RedRobin_df.iloc[16:37,5] = RedRobin_df.iloc[16:37,4]

temp_string = RedRobin_df.iloc[16:37,3]
temp_string = temp_string.str.split(" ", expand=True)
RedRobin_df.iloc[16:37,3] = temp_string.iloc[:,0]
RedRobin_df.iloc[16:37,4] = temp_string.iloc[:,1]

RedRobin_df.iloc[37:44,9] = RedRobin_df.iloc[37:44,8]
RedRobin_df.iloc[37:44,8] = RedRobin_df.iloc[37:44,7]
RedRobin_df.iloc[37:44,7] = RedRobin_df.iloc[37:44,6]
RedRobin_df.iloc[37:44,6] = RedRobin_df.iloc[37:44,5]
RedRobin_df.iloc[37:44,5] = RedRobin_df.iloc[37:44,4]
RedRobin_df.iloc[37:44,4] = RedRobin_df.iloc[37:44,3]

temp_string = RedRobin_df.iloc[37:44,2]
temp_string = temp_string.str.split(" ", expand=True)
RedRobin_df.iloc[37:44,2] = temp_string.iloc[:,0]
RedRobin_df.iloc[37:44,3] = temp_string.iloc[:,1]
RedRobin_df.insert(0,"Category", "TBD")
RedRobin_df.insert(0,"Resturant", "RedRobin")
RedRobin_df.columns = ["Resturant", "Category", "Item", "ServingSize", "Calories","CaloriesFromFat","TotalFat","SaturatedFat","TransFat","Cholesterol","Sodium","Carbohydrates","DietaryFiber","Sugar","Protien"]
RedRobin_df = RedRobin_df.drop([71,72],0)

test_df = RedRobin_df['Item'].apply(lambda x: CategoryFinderRR(x))
test_df = test_df.apply(pd.Series).stack().reset_index(drop = True) 
RedRobin_df['Category'] = test_df

InNOut = read_pdf("InNOut.pdf",pages='all',multiple_tables = True, pandas_options={'header': None})
InNOut_df = InNOut[0]
InNOut_df = InNOut_df.drop(23,1)
InNOut_df = InNOut_df.dropna(0)
InNOut_df = InNOut_df.rename(columns={0:"Item",1:"ServingSize",2:"Calories",3:"CaloriesFromFat",4:"TotalFat",5:"TF%OfDailyValue",6:"SaturatedFat",7:"SF%OfDailyValue",8:"TransFat",9:"Cholesterol",10:"Chol%OfDailyValue",11:"Sodium",12:"S%OfDailyValue",13:"Carbohydrates",14:"Carb%OfDailyValue",15:"DietaryFiber",16:"Fiber%OfDailyValue",17:"Sugar",18:"Protien",19:"%VitaminA",20:"%VitaminC",21:"%Calcium",22:"%Iron"})
InNOut_df.insert(0,"Category", "TBD")
InNOut_df.insert(0,"Resturant", "InNOut")

test_df = InNOut_df['Item'].apply(lambda x: CategoryFinder(x))
test_df = test_df.apply(pd.Series).stack().reset_index(drop = True) 

InNOut_df = InNOut_df.reset_index()
InNOut_df = InNOut_df.drop('index',1)
InNOut_df['Category'] = test_df

McDonalds = pd.read_csv("McDonalds.csv")
McDonalds.insert(0,"Resturant", "McDonalds")
McDonalds.columns = ["Resturant","Category","Item","ServingSize","Calories","CaloriesFromFat","TotalFat","TF%OfDailyValue","SaturatedFat","SF%OfDailyValue","TransFat","Cholesterol","Chol%OfDailyValue","Sodium","S%OfDailyValue","Carbohydrates","Carb%OfDailyValue","DietaryFiber","Fiber%OfDailyValue","Sugar","Protien","%VitaminA","%VitaminC","%Calcium","%Iron"]

Total_df = McDonalds.append(InNOut_df)
Total_df = Total_df.append(RedRobin_df)
Total_df = Total_df.reset_index()
Total_df = Total_df.drop('index',1)

Total_df.iloc[:,4:26] = Total_df.iloc[:,4:26].apply(pd.to_numeric)
Final_df = Total_df
Final_df['Category'] = Total_df['Category'].astype('category')
Final_df['Item'] = Total_df['Item'].astype('category')
Final_df['ServingSize'] = Total_df['ServingSize'].astype('category')
Final_df['Resturant'] = Total_df['Resturant'].astype('category')

dummy_df = pd.get_dummies(Final_df, columns=["Category","Resturant"])

dummy_df.to_csv("SuperSizeMe.csv")

#MODEL STOOF

imputer = KNNImputer(n_neighbors=2)
#dummy_df.iloc[:,3:36] = imputer.fit_transform(dummy_df.iloc[:,3:36])

#X = dummy_df.iloc[:,3:36]
#y = dummy_df['Calories']

Final_df.iloc[:,4:25] = imputer.fit_transform(Final_df.iloc[:,4:25])
Final_df = pd.get_dummies(Final_df, columns=["Resturant"])

X = Final_df.iloc[:,np.r_[3:27]]
y = Final_df['Category']

le = LabelEncoder() 
y = le.fit_transform(Final_df['Category'])

scaler = StandardScaler()
X = scaler.fit_transform(X)

#sns.pairplot(dummy_df)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=713)

#Naive Bayes
from sklearn.naive_bayes import MultinomialNB
nb = MultinomialNB()
nb.fit(X_train, y_train)
yhat = nb.predict(X_test)
print()
print("Naive Bayes")
print("Accuracy Score :",accuracy_score(y_test, yhat))
print("F1 Score :",f1_score(y_test, yhat, average='micro'))
print("Precision Score : ",precision_score(y_test, yhat, pos_label='positive', average='micro'))
print("Recall Score : ",recall_score(y_test, yhat, pos_label='positive',average='micro'))

#Decision Tree
from sklearn.tree import DecisionTreeClassifier

clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)
yhat_clf = clf.predict(X_test)

print()
print("Decision Tree")

print("Accuracy Score :",accuracy_score(y_test, yhat_clf))
print("F1 Score :",f1_score(y_test, yhat_clf, average='micro'))
print("Precision Score : ",precision_score(y_test, yhat_clf, pos_label='positive', average='micro'))
print("Recall Score : ",recall_score(y_test, yhat_clf, pos_label='positive',average='micro'))


#Random Forest
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier()
rf.fit(X_train, y_train)
yhat_rf = rf.predict(X_test)

print()
print("Random Forest")
print("Accuracy Score :",accuracy_score(y_test, yhat_rf))
print("F1 Score :",f1_score(y_test, yhat_rf, average='micro'))
print("Precision Score : ",precision_score(y_test, yhat_rf, average='micro'))
print("Recall Score : ",recall_score(y_test, yhat_rf,average='micro'))

#Gradient Boosting
from sklearn.ensemble import GradientBoostingClassifier

gb = GradientBoostingClassifier()
gb.fit(X_train, y_train)
yhat_gb = gb.predict(X_test)

print()
print("Gradient Boosting")
print("Accuracy Score :",accuracy_score(y_test, yhat_gb))
print("F1 Score :",f1_score(y_test, yhat_gb, average='micro'))
print("Precision Score : ",precision_score(y_test, yhat_gb, average='micro'))
print("Recall Score : ",recall_score(y_test, yhat_gb,average='micro'))

#Linear Regression
from sklearn.linear_model import LogisticRegression
print()
print("Fitting Logistic Regression")
lr = LogisticRegression(max_iter=10000)
lr.fit(X_train, y_train)

yhat_lr = lr.predict(X_test)

print("Accuracy Score :",accuracy_score(y_test, yhat_lr))
print("F1 Score :",f1_score(y_test, yhat_lr, average='micro'))
print("Precision Score : ",precision_score(y_test, yhat_lr, average='micro'))
print("Recall Score : ",recall_score(y_test, yhat_lr,average='micro'))


#KNN
print()
print("Fitting KNN")
knn = KNeighborsClassifier(1)
knn.fit(X_train, y_train)

yhat_knn = knn.predict(X_test)

print("Accuracy Score :",accuracy_score(y_test, yhat_knn))
print("F1 Score :", f1_score(y_test, yhat_knn, average='micro'))
print("Precision Score : ",precision_score(y_test, yhat_knn, average='micro'))
print("Recall Score : ",recall_score(y_test, yhat_knn,average='micro'))


#SVC
from sklearn.svm import SVC
print()
print("Fitting SVC")
svc = SVC(kernel='linear',probability=True)
svc.fit(X_train, y_train)

yhat_svc = svc.predict(X_test)

print("Accuracy Score :",accuracy_score(y_test, yhat_svc))
print("F1 Score :",f1_score(y_test, yhat_svc, average='micro'))
print("Precision Score : ",precision_score(y_test, yhat_svc, average='micro'))
print("Recall Score : ",recall_score(y_test, yhat_svc ,average='micro'))

#McDonalds - http://nutrition.mcdonalds.com/nutrition1/nutritionfacts.pdf
#In n Out - https://www.in-n-out.com/pdf/nutrition_2010.pdf
#Red Robin - https://www.redrobin.com/pages/nutrition/
#5 Guys - https://www.fiveguys.com/-/media/Public-Site/Files/FiveGuysNutrition_Aug2014_CAN_E.ashx
#Smash Burger - https://smashburger.com/wp-content/uploads/2018/05/Nutritional-Information-Smashburger-5-18.pdf
#Smash = read_pdf("SmashBurger.pdf",pages='all',multiple_tables = True, pandas_options={'header': None})


#Smash_df = Smash[1].append(Smash[2:10])
#Smash_df = Smash_df.drop([0,14],1)
#Smash_df = Smash_df.drop([0,1,2,3],0)
#Smash_df = Smash_df.reset_index()
#Smash_df = Smash_df.drop(['index'],1)
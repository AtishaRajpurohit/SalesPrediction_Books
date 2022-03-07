#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  4 14:48:07 2022

@author: atisharajpurohit
"""

"""Approach to answer the problem - Will we be able to both pay back our loan and afford our next book purchase order?

The question will be Yes/No depending on whether the total costs </> the total revenue

Constructing the equation of value

Total Costs :
    t_org: Total value of original books purchased+
    t_new: Total value of new books planning to be purchased+
    0.6$ * num_t_last: number of total books sent for last assortment+
    0.6$ * num_np_last: number of books NOT PURCHASED for last assortment+
    0.6$ * num_t_next: number of total books sent for next assortment +
    0.6$ * num_np_next: number of books NOT PURCHASED for next assortment (Needs to be assessed using an ML model)+
    
Total Revenue : 
    t_p_last: Total value of the number of books PURCHASED for last assortment+
    t_p_next: Total value of the number of books PURCHASED for next assortment (Needs to be assessed using an ML model)

"""
#Importing the required libraries
import pandas as pd
import numpy as np
import math
import os
import glob
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn import tree
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
import seaborn as sns
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score,KFold
from copy import deepcopy

#Aid in viewing tables completel
pd.set_option('display.max_columns',None)

#Reading in all the files as per their name
files=os.listdir('assessment/')#How to make this list without csv
files=[file for file in files if file.endswith('.csv')]
        
for n in files:
    x = n.replace(".csv" ,"")
    globals()[f"{x}"] = pd.read_csv('assessment/{}'.format(n),encoding="utf8")
    
#Checking the unique number of customer ids, to ensure these are 12000 values
customer_features['customer_id'].nunique()#Good to go , 12000 unique values in this dataset

###############################Dataset- Customer Features ###########################################
customer_features.head()
#Age bucket has nan values
customer_features.age_bucket.isna().sum()#500 nan values, so they cannot be removed.

#Imputing the age_bucket values with the mode
mode_age=customer_features['age_bucket'].mode()
customer_features['age_bucket']=customer_features['age_bucket'].replace(math.nan,mode_age[0])

#Splitting the genre into unique values and then , pivoting the dataset
#Removing the [] from genres
customer_features['favorite_genres']=customer_features['favorite_genres'].apply(lambda element: element.replace('[','').replace(']',''))


#Spliting genres by columns- this allows each genre to get picked up a separate element
cf2_cols=[(f'Genre_{i+1}') for i in range(10)]
customer_features[cf2_cols]=customer_features.favorite_genres.str.split(",",expand=True)
customer_features.drop(['favorite_genres'],1,inplace=True)

###############################Dataset- Product Features ###########################################

#Checking the unique number of customer ids, to ensure these are 1000 values
product_features['product_id'].nunique()#Good to go , 1000 unique values in this dataset

fig, (ax1, ax2) = plt.subplots(ncols=2,sharey=True)
sns.countplot(x="genre", palette="flare", data=product_features,ax=ax1)
ax1.tick_params(axis='x', rotation=80,labelsize=10)
ax1.title.set_text('Genres:Products (Supply)')

sns.countplot(x="Genre_1", palette="flare", data=customer_features,ax=ax2)
ax2.tick_params(axis='x', rotation=80,labelsize=10)
ax2.title.set_text('1st choice Genre:Consumers (Demand)')
plt.show()

################################################## Preparing the data for the model #######################################

#Since the last_assortment and next_assortment need to be joined,and have label encoding applied to them it makes sense to create a function to achieve this merging


def dataset_prep(base_df):
    """Merges the input dataset with relevant datasets to get customer features and product features,after performing label encoding."""
    mappings=[]
    base_df=pd.merge(base_df,customer_features,on='customer_id',how='left')
    base_df=pd.merge(base_df,product_features,on='product_id',how='left')
    label_encoder=LabelEncoder()
    for i,col in enumerate(base_df):
        if base_df[col].dtype=='object':
            base_df[col]=label_encoder.fit_transform(np.array(base_df[col].astype(str)).reshape(-1,))
            mappings.append(dict(zip(label_encoder.classes_,range(1,len(label_encoder.classes_)+1))))
    return base_df


#In this part of the code, the customer_features and the product features dataset will be joined to the
#last month assortment dataset. This is essentially creating the training dataset, with the target variable as purchased and the features as a combination of customer and product features.
lmo = dataset_prep(last_month_assortment)
#Creating a copy to use for later changes
lmo_copy=deepcopy(lmo)

#Checking number of unique customer ids and product ids
lmo['customer_id'].nunique()
#Only 720 ids out of the 12000 customer ids, so for the first assortment, not all customers were sent books.

#So for next assortment, need to check if the same customer ids have books sent to.
lmo['product_id'].nunique()
#All products sent in the first assortment, so there are enough training points

#Saving the labels and then dropping them from the dataset
lmoLabels=lmo["purchased"]
lmoIDLabels=lmo[["purchased","customer_id","product_id"]]
lmo.drop(["purchased","customer_id","product_id"], axis=1,inplace=True)

################################################## Model Selection, along with parameter selection ##########################################
#RandomForest - Since the data is qualitative and mainly categorical , it makes sense to use decision trees, with
#an ensemble method and hence random forests. The below performs a Random Forest on the model for a number if parameters.
#To aid with the code being run quicker this has already been run, and the best parameters have been chosen for the model.
#The code has still been kept and commented out, incase it needs to be verified.

# model=RandomForestClassifier()
# parameters={"n_estimators":[5,10,50,100,250],
#             "max_depth":[2,4,8,16,32,None]}

# cv=GridSearchCV(model,parameters,cv=5)
# cv.fit(lmo,lmoLabels)

# def display(results):
#     print(f'Best parameters are: {results.best_params_}')
#     print("\n")
#     mean_score = results.cv_results_['mean_test_score']
#     std_score = results.cv_results_['std_test_score']
#     params = results.cv_results_['params']
#     for mean,std,params in zip(mean_score,std_score,params):
#         print(f'{round(mean,3)} + or -{round(std,3)} for the {params}')
        
# display(cv)

#The model with the highest accuracy is with max_depth=16 and n_estimators=250 , however that is only marginally greater than max_depth:8 and n_estimators:50.
#Hence those parameters are chosen , as the computational time is much lesser
model_best=RandomForestClassifier(max_depth=8,n_estimators=50)
model_best.fit(lmo,lmoLabels)

#Finally testing the accuracy of the above model using cross-validation , instead of the train-test split
kfold=KFold(n_splits=5)
score=cross_val_score(model_best,lmo,lmoLabels,cv=kfold)
#print("Cross Validation Scores are {}".format(score))
#print("Average Cross Validation score :{}".format(score.mean()))

############################# Assessing the other components in the equation of value#############################

################################## Original Purchase order and Next Purchase Order ###############################

def total_cost_book(base_df):
    """Ensure these are the datasets containing purchase information.
    Generating total cost and total revenue for each product_id"""
    base_df['total_cost']=base_df['quantity_purchased']*base_df['cost_to_buy']
    base_df['revenue']=base_df['quantity_purchased']*base_df['retail_value']
    return base_df

original_purchase_order=total_cost_book(original_purchase_order)
next_purchase_order=total_cost_book(next_purchase_order)

#Viewing the difference in the cost and retail value. To get a broad understanding of the profit margins
chart1=sns.lineplot(data=original_purchase_order.reset_index(),x="index",y="cost_to_buy")
chart2=sns.lineplot(data=original_purchase_order.reset_index(),x="index",y="retail_value")
plt.title('Cost vs Retail as per the original purchase order')
plt.xlabel('Index')
plt.ylabel('Cost / Retail Value per product')
plt.legend(labels=["Cost Value","Retail Value"])
plt.show()

#The plot above shows clearly that the profit margins are large, thiscould be good news for situations where the consumers value their choice of books more than the price of these books.

t_org=original_purchase_order['total_cost'].sum()
t_new=next_purchase_order['total_cost'].sum()

################################### Last Month Assortment ###############################################
num_t_last=len(lmo_copy.product_id)

#Creating a numeric target variable
lmo_copy['n_purchased']=np.where(lmo_copy['purchased']==True,1,0)
num_np_last=(lmo_copy.purchased[lmo_copy['purchased']==False].count())

#Joining the retail price_column from the original dataset
lmo_copy=pd.merge(lmo_copy[['customer_id','product_id','purchased','n_purchased']],original_purchase_order[['product_id','retail_value']],on='product_id',how='left')
t_p_last=(lmo_copy['n_purchased']*lmo_copy['retail_value']).sum()

################################### Next Month Assortment ###############################################
#Extracting the customer_features and product_features to this dataset
nmo=dataset_prep(next_month_assortment)
Nmo_ID=nmo[['customer_id','product_id']]
nmo.drop(["customer_id","product_id"], axis=1,inplace=True)

#Applying the model to this dataset and then appending the predicted Labels to this dataset to calculate the expected revenue
#model_best.fit(TrainLAD,TrainLabels)
Nmo_predict=model_best.predict(nmo)
nmo['purchased']=np.array(Nmo_predict)

num_t_next=len(nmo.purchased)
nmo['n_purchased']=np.where(nmo['purchased']==True,1,0)
num_np_next=(nmo.purchased[nmo['purchased']==False].count())

#Pasting the product id to the nmo dataset and customer id
nmo = pd.concat([nmo, Nmo_ID], axis=1, join='inner')
#Getting the retail value from the nmo_copy dataset by merging using on product_id
nmo=pd.merge(nmo[['customer_id','product_id','purchased','n_purchased']],next_purchase_order[['product_id','retail_value']],on='product_id',how='left')
t_p_next=(nmo['n_purchased']*nmo['retail_value']).sum()

########################################## FINALLY ANSWERING THE BUSINESS QUESTION AT HAND ###########################

"""Finally proceeding to computing the equation of value. Any equation of value is incomplete without the rate of interest.
There are 2 rates of interest here:
    Interest on the loan
    Interest on the revenue generated from the books
    
Assumptions : 
1. The loan was purchased one month ago at month 0, is payable in 6 months as a lumpsum.
2. The last month assortment was sent for month 0.
3. The last month assortment will be sent a month from the present, hence month 2.
4. The next purchase will be assumed to be made on month 6, when the loan is due.

Interest accruued on shipping costs have been ignored for materiality.    
It must be noted that other costs, such as inventory costs, cost of books being damaged when being sent back , etc have been ignored.
"""

loan_interest=int(input("Please enter the per annum interest rate at which the loan was taken.The interest amount must be as a number. For example for 13% interest, enter 13.\nThe assumption is that the loan needs to be paid back in 6 months, as a lumpsum."))
bank_interest=int(input("Please enter the per annum interest rate the revenue will earn in the bank.The interest amount must be as a number. For example for 5% interest, enter 5."))


total_cost=t_org*((1+loan_interest/100)**0.5)+t_new
+0.6*(num_t_last+num_np_last+num_t_next+num_np_next) #Since the question states 0.6$/book and not per batch shipped to a customer
total_revenue=t_p_last*((1+bank_interest/100)**0.5)+t_p_next*((1+bank_interest/100)**(4/12))

if total_cost<total_revenue:
    print("Yes, however it must be noted that a number of costs such as inventory,maintainance, etc have been ignored from these calculations.")
else:
    print("No")


# date: July 5, 2021
# name: Martine De Cock
# description: Training ML models on IDASH2021, Track 3 data

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold

##################################################################################

def preprocess(dirty_df):
  dirty_df = dirty_df.drop(['patient_id','cohort_type'], axis = 1)
  target_map = {u'1': 1, u'0': 0}
  dirty_df['__target__'] = dirty_df['cohort_flag'].map(str).map(target_map)
  dirty_df = dirty_df.drop(['cohort_flag'], axis = 1)
  clean_X = dirty_df.drop('__target__', axis=1)
  clean_y = np.array(dirty_df['__target__'])
  return clean_X, clean_y

def evaluateForest(ntrees, RFresults, X_train, y_train, X1_train, y1_train, X2_train, y2_train, X_test, y_test): 
  clf1 = RandomForestClassifier(n_estimators=ntrees,random_state=10000)
  clf2 = RandomForestClassifier(n_estimators=ntrees,random_state=10000)
  clf  = RandomForestClassifier(n_estimators=ntrees,random_state=10000)
  
  clf1.fit(X1_train, y1_train)
  accP1  = accuracy_score(y_test,clf1.predict(X_test))

  clf2.fit(X2_train, y2_train)
  accP2 = accuracy_score(y_test,clf2.predict(X_test))
    
  clf.fit(X_train, y_train)
  accALL = accuracy_score(y_test,clf.predict(X_test))
  
  # Merging of RF models  
  clf1.estimators_ += clf2.estimators_
  clf1.n_estimators = len(clf1.estimators_)
  accMERG = accuracy_score(y_test,clf1.predict(X_test))

  RFresults[i] = [accP1,accP2,accALL,accMERG]
  
  
##################################################################################

  
# Load the data
df1 = pd.read_csv('party_1.csv')
df2 = pd.read_csv('party_2.csv')

X1, y1 = preprocess(df1)
X2, y2 = preprocess(df2)


# For each method, these will hold 4 accuracy results for each of 5 folds
# (1) accuracy of model trained on data from P1
# (2) accuracy of model trained on data from P2
# (3) accuracy of model trained on data from P1 and from P2
# (4) accuracy of aggregation of model (1) and (2) from above
LRresults = np.zeros((5, 4))
RF50results = np.zeros((5, 4))
RF100results = np.zeros((5, 4))
RF200results = np.zeros((5, 4))
RF400results = np.zeros((5, 4))

kf1 = KFold(n_splits=5,shuffle = True,random_state = 42)
kf2 = KFold(n_splits=5,shuffle = True,random_state = 42)


i = 0
for result1,result2 in zip(kf1.split(X1,y1),kf2.split(X2,y2)):
  print("FOLD ", i+1)
  X1_train, X1_test = X1.iloc[result1[0]], X1.iloc[result1[1]]
  y1_train, y1_test = y1[result1[0]], y1[result1[1]]
  X2_train, X2_test = X2.iloc[result2[0]], X2.iloc[result2[1]]
  y2_train, y2_test = y2[result2[0]], y2[result2[1]]

  X_train = X1_train.append(X2_train)
  y_train = np.append(y1_train,y2_train)
  X_test = X1_test.append(X2_test)
  y_test = np.append(y1_test,y2_test)


  ########## Train and test logistic regression models #################

  clf1 = LogisticRegression(solver='liblinear',random_state=10000)
  clf2 = LogisticRegression(solver='liblinear',random_state=10000)
  clf = LogisticRegression(solver='liblinear',random_state=10000)
  
  clf1.fit(X1_train, y1_train)
  accP1  = accuracy_score(y_test,clf1.predict(X_test))
  
  clf2.fit(X2_train, y2_train)
  accP2 = accuracy_score(y_test,clf2.predict(X_test))
    
  clf.fit(X_train, y_train)
  accALL = accuracy_score(y_test,clf.predict(X_test))
  
  # Merging of LR models
  clf1.coef_ = (clf1.coef_ + clf2.coef_)/2
  clf1.intercept_ = (clf1.intercept_ + clf2.intercept_)/2
  accMERG = accuracy_score(y_test,clf1.predict(X_test))  

  LRresults[i] = [accP1,accP2,accALL,accMERG]
    

  ########## Train and test RF models #################################

  ntrees=50
  RFresults=RF50results
  evaluateForest(ntrees, RFresults, X_train, y_train, X1_train, y1_train, 
                         X2_train, y2_train, X_test, y_test) 
  
  ntrees=100
  RFresults=RF100results
  evaluateForest(ntrees, RFresults, X_train, y_train, X1_train, y1_train, 
                         X2_train, y2_train, X_test, y_test) 

  ntrees=200
  RFresults=RF200results
  evaluateForest(ntrees, RFresults, X_train, y_train, X1_train, y1_train, 
                         X2_train, y2_train, X_test, y_test) 

  ntrees=400
  RFresults=RF400results
  evaluateForest(ntrees, RFresults, X_train, y_train, X1_train, y1_train, 
                         X2_train, y2_train, X_test, y_test) 


  print("==========completed")
  i = i + 1

# Printing the averages over the 5 folds
print("          P1,   P2, All, P1&P2")
np.set_printoptions(precision=2)
print("LR:     ",np.mean(LRresults, axis=0))
print("RF-50:  ",np.mean(RF50results, axis=0))
print("RF-100: ",np.mean(RF100results, axis=0))
print("RF-200: ",np.mean(RF200results, axis=0))
print("RF-400: ",np.mean(RF400results, axis=0)) 
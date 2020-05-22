import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score,StratifiedKFold, GridSearchCV, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression, Lasso
from sklearn.tree import DecisionTreeClassifier
from imblearn.over_sampling import ADASYN,SMOTE
from imblearn.ensemble import BalancedBaggingClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.utils import resample
from sklearn.feature_selection import SelectFromModel
from imblearn.under_sampling import NeighbourhoodCleaningRule
import pickle


data_white=pd.read_csv('C:/Users/Tolis/Desktop/wine/wine-white.csv', sep=';')
#data_red=pd.read_csv('C:/Users/Tolis/Desktop/wine/wine-red.csv', sep=';')



#def apply_type_white(index):
#    for index in data_white.index:
#        return 1
    
#def apply_type_red(row):
#   for index in data_red.index:
#      return 0
    
#data_white['wine_type']=data_white.apply(apply_type_white, axis=1)   
#data_red['wine_type']=data_white.apply(apply_type_red, axis=1)   
    
#dataset=pd.concat([data_white,data_red])
dataset=data_white

   

target_mapping={3:0, 4:0, 5:0, 6:1, 7:1, 8:1, 9:1}
dataset['quality']=dataset['quality'].map(target_mapping)


X=dataset.drop('quality', axis=1)
y=dataset['quality']


X_train, X_test, y_train, y_test = train_test_split(X,y, random_state=0, test_size=0.15)

train_set=pd.concat([X_train,y_train],axis=1)
test_set=pd.concat([X_test,y_test],axis=1)




sc=StandardScaler()
scaled_X_train=sc.fit_transform(train_set.drop('quality',axis=1))
scaled_X_test=sc.transform(test_set.drop('quality',axis=1))


sm=SMOTE(random_state=0)
X_train_res, y_train_res= sm.fit_resample(scaled_X_train, y_train)

X_train_res=pd.DataFrame(X_train_res, columns=X_train.columns)

train_set=pd.concat([X_train_res, y_train_res],axis=1)

scaled_X_test=pd.DataFrame(scaled_X_test, columns=X_test.columns)

test_set= pd.concat([scaled_X_test, test_set['quality'].reset_index(drop=True)],axis=1)


train_set.to_csv('train_set.csv', index=False)
test_set.to_csv('test_set.csv', index=False)

scaler='scaler.pkl'
file=open(scaler, 'wb')

pickle.dump(sc, file)

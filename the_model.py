import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score,StratifiedKFold, GridSearchCV, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression, Lasso
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import BorderlineSMOTE
from imblearn.ensemble import BalancedBaggingClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
from sklearn.feature_selection import SelectFromModel
import xgboost as xgb
import pickle

data=pd.read_csv('final_train_set.csv')
test_set=pd.read_csv('test_set.csv')
#test_set.drop('wine_type',axis=1,inplace=True)


X_train=data.drop(['quality'],axis=1)
y_train=data['quality']

X_test=test_set.drop(['quality'],axis=1)
y_test=test_set['quality']


rf=RandomForestClassifier()
rf.fit(X_train, y_train)
y_rf=rf.predict(X_test)

score_rf=cross_val_score(rf, X_train, y_train, cv=5, n_jobs=-1)
score_rf.mean()

n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt', 'log2']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10,15,20]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2,5,15,20]
# Method of selecting samples for training each tree
bootstrap = [True, False]# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}




Kfold=StratifiedKFold()
random=RandomizedSearchCV(rf ,param_distributions = random_grid, cv=Kfold, scoring="accuracy", n_jobs=-1, verbose = 1)
random.fit(X_train, y_train)
random.best_score_



rf_best=random.best_estimator_
rf_best.fit(X_train, y_train)
y_pred_bestrf=rf_best.predict(X_test)


print(classification_report(y_test,y_rf))
print(classification_report(y_test, y_pred_bestrf))



model='model.pkl'

file=open(model, 'wb')
pickle.dump(rf_best, file)



















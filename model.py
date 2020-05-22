import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score,StratifiedKFold, GridSearchCV, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression, Lasso
from sklearn.tree import DecisionTreeClassifier
from imblearn.over_sampling import SMOTE
from imblearn.ensemble import BalancedBaggingClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
from sklearn.feature_selection import SelectFromModel

skf = StratifiedKFold(n_splits=10)


data_white=pd.read_csv('C:/Users/Tolis/Desktop/wine/wine-white.csv', sep=';')
data_red=pd.read_csv('C:/Users/Tolis/Desktop/wine/wine-red.csv', sep=';')



def apply_type_white(index):
    for index in data_white.index:
        return 1
    
def apply_type_red(row):
    for index in data_red.index:
        return 0
    
data_white['wine_type']=data_white.apply(apply_type_white, axis=1)   
data_red['wine_type']=data_white.apply(apply_type_red, axis=1)   
    
dataset=pd.concat([data_white,data_red])

   

target_mapping={3:0, 4:0, 5:0, 6:1, 7:1, 8:1, 9:1}
dataset['quality']=dataset['quality'].map(target_mapping)


X=dataset.drop('quality', axis=1)
y=dataset['quality']


X_train, X_test, y_train, y_test = train_test_split(X,y, random_state=0, test_size=0.2)




train_set=pd.concat([X_train,y_train],axis=1)

majority_class=train_set[train_set['quality']==1]

minority_class=train_set[train_set['quality']==0]



minority_upsampled=resample(minority_class, replace=True, n_samples=3107,random_state=0)


train_set=pd.concat([majority_class, minority_upsampled])

X_train=train_set.drop('quality',axis=1)
y_train=train_set['quality'] 






X_traint, X_val, y_traint, y_val = train_test_split(X_train, y_train, test_size=0.33 , random_state=0)


feature_sel_model = SelectFromModel(Lasso(alpha=0.001, random_state=0)) # remember to set the seed, the random state in this function
feature_sel_model.fit(X_traint, y_traint)

feature_sel_model = RFECV(Lasso(alpha=0.001, random_state=0), cv=5)
feature_sel_model.fit(X_train, y_train)



feature_sel_model.get_support()


# let's print the number of total and selected features

# this is how we can make a list of the selected features
selected_feat = X_traint.columns[(feature_sel_model.get_support())]

# let's print some stats
print('total features: {}'.format((X_traint.shape[1])))
print('selected features: {}'.format(len(selected_feat)))


score=cross_val_score(SelectFromModel, X_train,y_test, cv=5)














#Create an object of the classifier.
bbc = BalancedBaggingClassifier(base_estimator=DecisionTreeClassifier(),
                                sampling_strategy='auto',
                                replacement=False,
                                random_state=0)


#Train the classifier.
bbc.fit(X, y)
y_train_new = bbc.predict(X_train)

smote=SMOTE(random_state=0)

X_train_res, y_train_res= smote.fit_resample(X,y)

sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.transform(X_test)

log_model=LogisticRegression()
log_model.fit(X_train, y_train)
y_pred_log=log_model.predict(X_test)

score=cross_val_score(log_model,X_train, y_train_new, cv=skf)


from sklearn.ensemble import RandomForestClassifier

rf=RandomForestClassifier()
rf.fit(X_train, y_train)
y_pred_rf=rf.predict(X_test)




n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]
# Method of selecting samples for training each tree
bootstrap = [True, False]# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}



random=RandomizedSearchCV(rf ,param_distributions = random_grid, cv=5, scoring="accuracy", n_jobs=-1, verbose = 1)
random.fit(X_train, y_train)

rf_best=random.best_estimator_
rf_best.fit(X_train, y_train)
y_pred_bestrf=rf_best.predict(X_test)



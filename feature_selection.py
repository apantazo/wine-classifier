import pandas as pd
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression, Lasso, LassoCV
from sklearn.model_selection import train_test_split
import statsmodels.api as sm

train_set=pd.read_csv('train_set.csv')


X=train_set.drop('quality', axis=1)
y=train_set['quality']




X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15 , random_state=0)





feature_sel_model = SelectFromModel(Lasso(alpha=0.001, random_state=0)) # remember to set the seed, the random state in this function
feature_sel_model.fit(X_train, y_train )

feature_sel_model.get_support()


# let's print the number of total and selected features

# this is how we can make a list of the selected features
selected_feat = X_train.columns[(feature_sel_model.get_support())]

# let's print some stats
print('total features: {}'.format((X_train.shape[1])))
print('selected features: {}'.format(len(selected_feat)))




from sklearn.ensemble import ExtraTreesClassifier
import matplotlib.pyplot as plt
model = ExtraTreesClassifier()
model.fit(X_train, y_train)

feat_importances = pd.Series(model.feature_importances_, index=X_train.columns)
feat_importances.nlargest(X_train.shape[1]).plot(kind='barh')
plt.show()

#X_features=X.drop(['wine_type'],axis=1)

X_features=X

y_features=y

final_train_set=pd.concat([X_features,y_features],axis=1)

final_train_set.to_csv('final_train_set.csv', index=False)



import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Imputer


#make the file easier to access
titanic_file_path = '/Users/benjaminmcdole/Desktop/train.csv'

titanic_test_path = '/Users/benjaminmcdole/Desktop/test.csv'
#read the data and store it for use
titanic_data = pd.read_csv(titanic_file_path)
titanic_test_data = pd.read_csv(titanic_test_path)

#print a quick summary
#print(titanic_data.describe())
#print(titanic_test_data.describe())

#list column names
#print(titanic_data.columns)

#print the first few rows, get an idea of the info
#print(titanic_data.head(5))
#print(titanic_test_data.head(5))

#set what we want predicted
y = titanic_data.Survived

#choose predictors (blank for now, because who knows what they'll be)
titanic_data_predictors = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']

#by convention we call predictors X
X = titanic_data[titanic_data_predictors]
test_X = titanic_test_data[titanic_data_predictors]

#one-hot encode the categorical stuff
encoded_X = pd.get_dummies(X)
encoded_test_X = pd.get_dummies(test_X)

#fix missing numeric values
my_imputer = Imputer()
imputed_X = pd.DataFrame(my_imputer.fit_transform(encoded_X))
imputed_encoded_test_X = pd.DataFrame(my_imputer.fit_transform(encoded_test_X))

#split data
train_X, val_X, train_y, val_y = train_test_split(imputed_X, y, random_state = 0)

#try second model randomforest
titanic_random_forest_model = RandomForestRegressor()
titanic_random_forest_model.fit(train_X, train_y)
test_preds2 = titanic_random_forest_model.predict(imputed_encoded_test_X)
rounded_preds2 = np.round(test_preds2)
#print(mean_absolute_error(val_preds2, y))

my_submission2 = pd.DataFrame({'PassengerId': titanic_test_data.PassengerId, 'Survived': rounded_preds2})
my_submission2.to_csv('/Users/benjaminmcdole/Desktop/submission3.csv', index=False)
#my_comparison = pd.DataFrame({'Test solutions': val_y, 'Predicted solutions': val_preds3})
#print(my_comparison)

'''
#try first model, decision tree
#define model
titanic_decision_tree_model = DecisionTreeRegressor()

#fit model
titanic_decision_tree_model.fit(train_X, train_y)

#get predicted prices on validation data, print MAE
val_preds1 = titanic_decision_tree_model.predict(val_X)
print(mean_absolute_error(val_y, val_preds1))

#try third model Logisticregression
#probably best for binary outcome
titanic_logistic_regression_model = LogisticRegression()
titanic_logistic_regression_model.fit(train_X, train_y)
val_preds3 = titanic_logistic_regression_model.predict(val_X)
test_preds3 = titanic_logistic_regression_model.predict(imputed_encoded_test_X)
'''
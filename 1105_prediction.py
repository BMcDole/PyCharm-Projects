import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import LogisticRegression


#make the file easier to access
1105_file_path = ''

#read the data and store it for use
1105_data = pd.read_csv(1105_file_path)

#print a quick summary
print(1105_data.describe())

#list column names
print(1105_data.columns)

'''
#set what we want predicted
y = 1105_data.Grade

#choose predictors (blank for now, because who knows what they'll be)
1105_data_predictors = [column names here]

#by convention we call predictors X
X = 1105_data[1105_data_predictors]

#try first model, decision tree
#define model
1105_decision_tree_model = DecisionTreeRegressor()

#split data
train_X, val_X, train_y, vay_y = train_test_split(X, y, random_state = 0)

#fit model
1105_decision_tree_model.fit(train_X, train_y)

#get predicted prices on validation data, print MAE
val_preds1 = 1105_decision_tree_model.predict(val_X)
print(mean_absolute_error(val_y, val_preds1))

#try second model randomforest
1105_random_forest_model = RandomForestRegressor()
1105_random_forest_model.fit(train_X, train_y)
val_preds2 = 1105_random_forest_model.predict(val_X)
print(mean_absolute_error(val_y, val_preds2)

#try third model Logisticregression
1105_logistic_regression_model = LogisticRegression()
1105_logistic_regression_model.fit(train_X, train_y)
val_preds3 = 1105_logistic_regression_model.predict(val_X)
print(mean_absolute_error(val_y, val_preds3)


'''
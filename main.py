# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.


import pandas
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score, roc_auc_score, make_scorer
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix
import matplotlib


Bankruptcy_Data = pandas.read_excel('/Users/carstenjuliansavage/Desktop/R Working Directory/MasterDataforML.xlsx')
pandas.set_option('display.max_columns', None)

# Filtering dataset for input and output variables only

Bankruptcy_Data_Slim = (Bankruptcy_Data
    .filter(['cash_debt', 'curr_debt', 'int_totdebt', 'quick_ratio', 'de_ratio', 'debt_assets', 'intcov','isBankrupt'])
    #.dropna()
)

X = Bankruptcy_Data_Slim[['cash_debt',
                          'curr_debt',
                          'int_totdebt',
                          'quick_ratio',
                          'de_ratio',
                          'debt_assets',
                          'intcov']]
y = Bankruptcy_Data_Slim[['isBankrupt']]

# Scaling the data to be between 0 and 1
min_max_scaler = preprocessing.MinMaxScaler()
X = min_max_scaler.fit_transform(X)
y = min_max_scaler.fit_transform(y)

# Split dataframe into training and testing data. Remember to set a seed.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=47)

# Let's confirm that the scaling worked as intended.
# All values should be between 0 and 1 for all variables.
X_Stats = pandas.DataFrame(X)
X_Stats.describe()

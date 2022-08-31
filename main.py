import pandas
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score, roc_auc_score, make_scorer
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix
import matplotlib
from sklearn import preprocessing


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

# Split dataframe into training and testing data. Remember to set a seed and use stratification.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=47, stratify=y)

# Let's confirm that the scaling worked as intended.
# All values should be between 0 and 1 for all variables.
X_Stats = pandas.DataFrame(X)
X_Stats.describe()

sum(y_train)/len(y_train)
sum(y_test)/len(y_test)

XGB = xgb.XGBClassifier(objective='binary:logistic',
                            missing=0,
                            seed=47)
XGB.fit(X_train,
        y_train,
        verbose=True,
        early_stopping_rounds=15,
        eval_metric = 'aucpr',
        eval_set=[(X_test,y_test)])

plot_confusion_matrix(XGB,
                      X_test,
                      y_test,
                      values_format='d',
                      display_labels=["Not Bankrupt","Bankrupt"])

# Let's attempt to impose model error costs on the Bankrupt obs.

# Let's optimize the parameters - First Pass.
To_Optimize_Parameters = {
    'max_depth':[1,2,3,4,5],
    'learning_rate':[1.0,0.1,0.01,0.001],
    'gamma':[0,0.5,1.0],
    'reg_lambda':[0,1.0,10.0],
    'scale_pos_weight':[1,3,5]
}

# It appears that reg_lambda and scale_pos_weight are signaling that the optimal value is higher.
# Let's rerun the parameter optimization with a higher upper bound of possible values.
To_Optimize_Parameters = {
    'max_depth':[4],
    'learning_rate':[0.1],
    'gamma':[0.5],
    'reg_lambda':[10.0,20.0,100.0],
    'scale_pos_weight':[5.0,10.0,15.0]
}

# Final parameter optimization pass. Just need to refine reg_lambda.
To_Optimize_Parameters = {
    'max_depth':[4],
    'learning_rate':[0.1],
    'gamma':[0.5],
    'reg_lambda':[100.0, 250.0, 500.0, 750.0, 1000.0],
    'scale_pos_weight':[5.0]
}

optimal_params = GridSearchCV(
    estimator=xgb.XGBClassifier(objective='binary:logistic',
                                seed=47,
                                subsample=0.9,
                                colsample_bytree=0.5),
    param_grid = To_Optimize_Parameters,
    scoring = 'roc_auc',
    verbose = 0,
    n_jobs = 10,
    cv = 3
)

optimal_params.fit(X_train,
                   y_train,
                   early_stopping_rounds=15,
                   eval_metric='auc',
                   eval_set=[(X_test,y_test)],
                   verbose=False)

print(optimal_params.best_params_)


XGB_Refined = xgb.XGBClassifier(seed = 47,
                                objective='binary:logistic',
                                gamma=0.5,
                                learn_rate=0.1,
                                max_depth=4,
                                reg_lambda=100.0,
                                scale_pos_weights=5.0,
                                subsample=0.9,
                                colsample_bytree=0.5)

XGB_Refined.fit(X_train,
                y_train,
                verbose=True,
                early_stopping_rounds=15,
                eval_metric='aucpr',
                eval_set=[(X_test,y_test)])

plot_confusion_matrix(XGB_Refined,
                      X_test,
                      y_test,
                      values_format='d',
                      display_labels=["Not Bankrupt","Bankrupt"])
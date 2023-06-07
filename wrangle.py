import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore")
from sklearn.cluster import KMeans
import scipy.stats as stats
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
import os
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import QuantileTransformer
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import SelectKBest, RFE, f_regression, SequentialFeatureSelector
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.linear_model import LassoLars
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import TweedieRegressor

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier


def split_data(df, target):
    '''
    Takes in a dataframe and returns train, validate, test subset dataframes
    '''
    
    
    train, test = train_test_split(df,
                                   test_size=.2, 
                                   random_state=123, 
                                   stratify = df[target]
                                   )
    train, validate = train_test_split(train, 
                                       test_size=.25, 
                                       random_state=123, 
                                       stratify = train[target]
                                       )
    
    return train, validate, test


def prepare_data(df):
    df = df.drop(columns = ['CholCheck', 'PhysActivity', 'AnyHealthcare', 'NoDocbcCost','DiffWalk','GenHlth','Fruits','Veggies','Income','Education','Stroke'])

    df.columns = df.columns.str.lower()

    df['age'][df['age'] == 1] = '18 to 24'
    df['age'][df['age'] == 2] = '25 to 29'
    df['age'][df['age'] == 3] = '30 to 34'
    df['age'][df['age'] == 4] = '35 to 39'
    df['age'][df['age'] == 5] = '40 to 44'
    df['age'][df['age'] == 6] = '45 to 49'
    df['age'][df['age'] == 7] = '50 to 54'
    df['age'][df['age'] == 8] = '55 to 59'
    df['age'][df['age'] == 9] = '60 to 64'
    df['age'][df['age'] == 10] = '65 to 69'
    df['age'][df['age'] == 11] = '70 to 74'
    df['age'][df['age'] == 12] = '75 to 79'
    df['age'][df['age'] == 13] = '80 or older'


    df['sex'][df['sex'] == 1] = 'male'
    df['sex'][df['sex'] == 0] = 'female'
    df['bmi'] = df['bmi'][df['bmi'] < df['bmi'].quantile(.99)].copy()
    df = df.dropna()
    return df


def chi2_test(train, target, columns_list):
    '''
    Runs a chi2 test on all items in a list of lists and returns a pandas dataframe
    '''
    chi_df = pd.DataFrame({'feature': [],
                    'chi2': [],
                    'p': [],
                    'degf':[],
                    'expected':[]})
    
    for iteration, col in enumerate(columns_list):
        
        observed = pd.crosstab(train[target], train[col])
        chi2, p, degf, expected = stats.chi2_contingency(observed)

        chi_df.loc[iteration+1] = [col, chi2, p, degf, expected]

    return chi_df


def create_knn(X_train,y_train, X_validate, y_validate):
    '''
    creating a logistic_regression model
    fitting the logistic_regression model
    predicting the training and validate data
    '''
    the_df = pd.DataFrame(data=[
    {
        'model_train':'knn',
        'train_predict':229787/(229787+23893),
        'validate_predict':229787/(229787+23893),
        'n_neighbors': 'neighbors'
    }
    ])
    for i in range(20):
        knn = KNeighborsClassifier(n_neighbors=i+1)
        knn.fit(X_train, y_train)
        train_predict = knn.score(X_train, y_train)
        validate_predict = knn.score(X_validate, y_validate)
        the_df.loc[i+1] = ['KNeighborsClassifier', train_predict, validate_predict, i+1]

    return the_df


def create_logistic_regression(X_train,y_train, X_validate, y_validate):
    '''
    creating a logistic_regression model
    fitting the logistic_regression model
    predicting the training and validate data
    '''
    the_df = pd.DataFrame(data=[
    {
        'model_train':'LogisticRegression',
        'train_predict':229787/(229787+23893),
        'validate_predict':229787/(229787+23893),
        'C': 'the_c'
    }
    ])

    for iteration, i in enumerate([.01, .1, 1, 10, 100, 1000]):
        logit = LogisticRegression(random_state= 123,C=i)
        logit.fit(X_train, y_train)
        train_predict = logit.score(X_train, y_train)
        validate_predict = logit.score(X_validate, y_validate)
        the_df.loc[iteration + 1] = ['LogisticRegression', train_predict, validate_predict, i]

    return the_df


def create_random_forest(X_train,y_train, X_validate, y_validate,X_test, y_test):
    '''
    creating a random_forest model
    fitting the random_forest model
    predicting the training and validate data
    '''
    the_df = pd.DataFrame(data=[
    {
        'model_train':'RandomForestClassifier',
        'train_predict':229787/(229787+23893),
        'validate_predict':229787/(229787+23893),
        'max_depth': 'max_depth'
    }
    ])
    test_df = pd.DataFrame(data=[
    {
        'model_train':'RandomForestClassifier',
        'baseline':229787/(229787+23893),
        'max_depth': 'max_depth'
    }
    ])

    for i in range(20):
        forest = RandomForestClassifier(random_state = 123,max_depth=i +1 )
        forest.fit(X_train, y_train)    
        train_predict = forest.score(X_train, y_train)
        validate_predict = forest.score(X_validate, y_validate)
        the_df.loc[i + 1] = ['RandomForestClassifier', train_predict, validate_predict, i + 1]

    forest = RandomForestClassifier(random_state = 123,max_depth=4 )
    forest.fit(X_train, y_train)  
    test_predict = forest.score(X_test, y_test)
    test_df.loc[1] = ['RandomForestClassifier', round(test_predict, 3), 16]
    
    return the_df, test_df


def create_descision_tree(X_train,y_train, X_validate, y_validate):
    '''
    creating a Decision tree model
    fitting the Descision tree model
    predicting the training and validate data
    '''

    the_df = pd.DataFrame(data=[
    {
        'model_train':'DecisionTreeClassifier',
        'train_predict':2255/(2255+1267),
        'validate_predict':2255/(2255+1267),
        'max_depth': 'max_depth'
    }
    ])

    for i in range(20):

        tree = DecisionTreeClassifier(random_state = 123,max_depth= i + 1)
        tree.fit(X_train, y_train)
        train_predict = tree.score(X_train, y_train)
        validate_predict = tree.score(X_validate, y_validate)
        the_df.loc[i + 1] = ['DecisionTreeClassifier', train_predict, validate_predict, i + 1]

    return the_df


def scale_data(train,
               validate,
               test,
               cols = ['alcohol', 'density']):
    '''Takes in train, validate, and test set, and outputs scaled versions of the columns that were sent in as dataframes'''
    #Make copies for scaling
    train_scaled = train.copy() #Ah, making a copy of the df and then overwriting the data in .transform() to remove warning message
    validate_scaled = validate.copy()
    test_scaled = test.copy()
    #Initiate scaler, using Robust Scaler
    scaler = MinMaxScaler()
    #Fit to train only
    scaler.fit(train[cols])
    #Creates scaled dataframes of train, validate, and test. This will still preserve columns that were not sent in initially.
    train_scaled[cols] = scaler.transform(train[cols])
    validate_scaled[cols] = scaler.transform(validate[cols])
    test_scaled[cols] = scaler.transform(test[cols])

    return train_scaled, validate_scaled, test_scaled


def mvp_info(train_scaled, validate_scaled, test_scaled,list_of_features, target):
    '''
    Takes in scaled data and a list of features to create the different feature and target variable objects
    '''
    X_train = train_scaled[list_of_features]
    X_validate = validate_scaled[list_of_features]
    X_test = test_scaled[list_of_features]


    y_train = train_scaled[target]
    y_validate = validate_scaled[target]
    y_test = test_scaled[target]

    return X_train, X_validate, X_test, y_train, y_validate, y_test


def get_dummies(X_train, X_validate, X_test,the_columns):
    dummy_train = pd.get_dummies(X_train[the_columns])
    dummy_validate = pd.get_dummies(X_validate[the_columns])
    dummy_test = pd.get_dummies(X_test[the_columns])

    X_train = pd.concat([X_train, dummy_train], axis=1)
    X_validate = pd.concat([X_validate, dummy_validate], axis=1)
    X_test = pd.concat([X_test, dummy_test], axis=1)

    X_train = X_train.drop(columns =['sex', 'age'])
    X_validate = X_validate.drop(columns =['sex', 'age'])
    X_test = X_test.drop(columns =['sex', 'age'])

    return X_train, X_validate, X_test


def get_second_list(df):
    the_list = list(df.columns)
    second_list = []
    target = the_list.pop(0)
    second_list.append(the_list.pop(-1))
    second_list.append(the_list.pop(2))
    second_list.append(the_list.pop(-3))
    second_list.append(the_list.pop(-2))
    the_age = second_list.pop(0)

    return second_list , the_age ,the_list, target


def calculate_percentage(value1, value2):
    the_df = pd.DataFrame(data=[
    {
        'No heart problems':value1,
        'Heart problems':value2,
        'Percent heart problems':(value2 / (value1 + value2) ) * 100,
    }
    ])
    return the_df


def combine_three_dataframes(df1, df2, df3):
    return pd.concat([df1, df2, df3])


def combine_two_dataframes(df1, df2):
    return pd.concat([df1, df2])

def the_order():
    the_order = ['18 to 24', '25 to 29', '30 to 34','35 to 39', '40 to 44',
            '45 to 49', '50 to 54', '55 to 59', '60 to 64', '65 to 69', 
            '70 to 74', '75 to 79', '80 or older']
    return the_order


def comparison_of_means(df, second_list):
    df1 = pd.DataFrame()
    for i in second_list:
        t, p = stats.ttest_ind(df[i][df.heartdiseaseorattack == 1.0],df[i][df.heartdiseaseorattack == 0.0])
        df1 = pd.concat([df1, pd.DataFrame(data =[
            {
                'Category name':i,
                'P value':p
            }
        ])], axis = 0)
    return df1




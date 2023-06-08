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


def create_random_forest(X_train,y_train, X_validate, y_validate,X_test, y_test):
    '''
    gets the weights of the best model
    '''
    
    forest = RandomForestClassifier(random_state = 123,max_depth=9 )
    forest.fit(X_train, y_train)    
    train_predict = forest.score(X_train, y_train)
    validate_predict = forest.score(X_validate, y_validate)
    the_weights = forest.feature_importances_
    
    return forest


def getting_weights_max(tree, X_train):
    columns = X_train.columns

    the_weight = tree.feature_importances_
    the_weight
    weights_column = []
    for i in the_weight:
        weights_column.append(i)
        
    the_dataframe = pd.DataFrame({'columns': columns, 
                                'the_weight':weights_column})  

    the_dataframe
    plt.title('Does device_protection affect churn')  

    ax = sns.barplot(x=columns , y=the_weight, data = the_dataframe)
    ax.tick_params(axis='x', rotation=90)
    plt.show()
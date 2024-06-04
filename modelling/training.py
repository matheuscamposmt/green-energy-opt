import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import GridSearchCV
from . import tuning
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, OneHotEncoder
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression, ElasticNet
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.impute import SimpleImputer
import numpy as np
from tqdm import tqdm
import os
from sklearn.model_selection import cross_validate
from sklearn.base import RegressorMixin
from sklearn.metrics import mean_squared_error, mean_absolute_error, make_scorer



from feature_engineering import cropland, district, elevation, fe_utils, rainfall

def melt(data, years=range(2010, 2018)):
    """Melt the data

    Args:
        data (pd.DataFrame): Dataframe
        years (list, optional): List of years. Defaults to range(2010, 2018).

    Returns:
        pd.DataFrame: Melted dataframe
    """

    years_columns = [str(year) for year in years]
    melted = pd.melt(data, id_vars=set(data.columns) - set(years_columns), value_vars=years_columns, var_name='year', value_name='biomass')
    melted['year'] = melted['year'].astype(int)

    return melted

def add_features(X, offset=0.05, years=range(2010, 2018)):
    """
    Adds additional features to the input data.

    Parameters:
    - X: The input data.
    - offset: The offset value for set_grid_blocks function (default: 0.05).

    Returns:
    The input data with additional features added.
    """
    t = tqdm(total=4, desc="Adding features")
    X = fe_utils.set_grid_blocks(X, offset=offset)
    t.update(1)
    t.set_description("Adding district features")
    X = district.add_district_feature(X)
    t.update(1)

    t.set_description("Adding elevation features")
    X = elevation.add_elevation_features(X)
    t.update(1)

    t.set_description("Adding cropland features")
    X = cropland.add_cropland_feature(X)
    t.update(1)
    
    X = melt(X, years=years)

    #t.set_description('Adding crop production features')
    #X = crop_production.add_crop_production_feature(X)
    #t.update(1)

    
    #t.set_description("Adding rainfall features")
    #X = rainfall.add_rainfall_feature(X, forecast_horizon=1, years=years)
    #t.update(1)

    t.set_description("Finished")

    t.close()

    # reconvert from geodataframe to pd.dataframe
    X = X.drop(columns=['block'])
    X = pd.DataFrame(X)

    return X


preprocessors = {
    'linear_regression': {'num': StandardScaler(), 'cat': OneHotEncoder()},
    'random_forest': {'num': 'passthrough', 'cat': OrdinalEncoder()},
    'xgboost': {'num': 'passthrough', 'cat': OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)},
    'decision_tree': {'num': 'passthrough', 'cat': OrdinalEncoder()},
    'extra_trees':{'num': 'passthrough', 'cat': OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)},
    'lightgbm': {'num': 'passthrough', 'cat': OrdinalEncoder()}
}

def get_preprocessor(df_X: pd.DataFrame, algorithm: str | RegressorMixin):

    numeric_columns = df_X.select_dtypes(include='number').columns.tolist()
    categorical_columns = df_X.select_dtypes(include='object').columns.tolist()

    if isinstance(algorithm, str):
        print(f"Algorithm: {algorithm}")
        algorithm_name = algorithm
    elif isinstance(algorithm, LinearRegression) or isinstance(algorithm, ElasticNet):
        algorithm_name = 'linear_regression'
    elif isinstance(algorithm, RandomForestRegressor):
        algorithm_name = 'random_forest'
    elif isinstance(algorithm, XGBRegressor):
        algorithm_name = 'xgboost'
    elif isinstance(algorithm, ExtraTreesRegressor):
        algorithm_name = 'extra_trees'
    elif isinstance(algorithm, DecisionTreeRegressor):
        algorithm_name = 'decision_tree'
    elif isinstance(algorithm, LGBMRegressor):
        algorithm_name = 'lightgbm'
    else:
        raise ValueError(f"Algorithm {algorithm} not included in the preprocessors")

    numerical_transformer = preprocessors[algorithm_name]['num']
    categorical_transformer = preprocessors[algorithm_name]['cat']    

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numeric_columns),
            ('cat', categorical_transformer, categorical_columns)
        ]
    )


    return preprocessor

class TimeSeriesSplit:
    def __init__(self, sliding_windows=False, window_size=1):
        self.sliding_windows = sliding_windows
        self.window_size = window_size
    
    def split(self, X, y=None, groups=None):
        n = len(X)
        step = self.window_size * 2418
        for i in range(0, n - step, 2418):
            train_index = np.arange(0 + i*int(self.sliding_windows), i + step)
            test_index = np.arange(i + step, i + step + 2418)
            yield train_index, test_index

    def get_n_splits(self, X, y=None, groups=None):
        return (len(X) // 2418) - self.window_size

def logging_results(results, features, filename='results.csv'):
    results = results.assign(timestamp=pd.Timestamp.now(), features=str(features))
    results.to_csv('modelling/logs/'+filename, mode='a', index=False, header=not os.path.exists('modelling/logs/'+filename))

def get_pipeline(algorithm, X_train, alg_type=None):

    pipe = Pipeline(
        steps=[
            ('preprocessor', get_preprocessor(X_train, alg_type if alg_type else algorithm)),
            ('imputer', SimpleImputer(strategy='median')),
            ('regressor', algorithm)
        ]
    )

    return pipe
def     model_selection_workflow(X_train, y_train, algorithms):
    """
    Perform model selection workflow for a given set of algorithms.

    Args:
        X_train (pandas.DataFrame): The input features for training.
        y_train (pandas.Series): The target variable for training.
        algorithms (dict): A dictionary of algorithms to be evaluated.

    Returns:
        pandas.DataFrame: The results of the model selection workflow.

    """
    scoring_metrics = {
        'rmse': make_scorer(mean_squared_error, squared=False),
        'mae': make_scorer(mean_absolute_error)
    }
    results = []
    i = 0
    #np.int = int
    for name, algorithm in algorithms.items():
        print(f"[{i}] Running \"{name}\" algorithm...")
        i += 1
        pipe = get_pipeline(algorithm, X_train)

        tscv = TimeSeriesSplit(sliding_windows=False, window_size=1)

        print(f"\t[i] Cross validating...")
        evaluation = cross_validate(pipe, X_train, y_train,
                                    cv=tscv, scoring=scoring_metrics, n_jobs=-1,
                                    return_train_score=True)
        
        print("\tGathering results...")
        scores = {
            'algorithm': name,
            'train_rmse': evaluation['train_rmse'].mean(),
            'train_mae': evaluation['train_mae'].mean(),
            'test_rmse': evaluation['test_rmse'].mean(),
            'test_mae': evaluation['test_mae'].mean(),
            'fit_time': evaluation['fit_time'].mean()
        }

        results.append(scores)

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values(by='test_mae')

    # write the results into a file adding a timestamp and concatenate with the previous results
    logging_results(results_df, features=X_train.columns.tolist())

    return results_df


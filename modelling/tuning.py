


# --- Hyperparameter Tuning ---
from sklearn.metrics import make_scorer, mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from xgboost import XGBRegressor
from sklearn.linear_model import ElasticNet, LinearRegression
from skopt import BayesSearchCV


hyperparameters = {
    'RandomForestRegressor': {
       'regressor__n_estimators': [20, 30, 50, 70, 100],
        'regressor__max_depth': [5, 8, 12, 15, 17]
    },
    'ElasticNet': {
        'regressor__alpha': [0.5, 1.0, 2.0],
        'regressor__l1_ratio': [0.5, 0.7, 0.9]
    },
    'XGBRegressor': {
        'regressor__n_estimators': [200, 300, 500, 700, 1000],
        'regressor__learning_rate': [0.03, 0.05, 0.1],
        'regressor__max_depth': [4, 6, 8, 12],
        'regressor__gamma': [0, 0.1, 0.2, 0.3]
    },
    'ExtraTreesRegressor': {
        'regressor__n_estimators': [20, 30, 50, 70, 100],
        'regressor__max_depth': [5, 8, 12, 15, 17]
    },

    'LinearRegression': {}
}



def tune_hyperparameters(X_train, y_train, algorithm):
    hyperparameters = get_hyperparameters(algorithm)

    scorers = {
        'MSE': make_scorer(mean_squared_error),
        'MAE': make_scorer(mean_absolute_error),
        'r2': make_scorer(r2_score),
    }

    bayes_search = BayesSearchCV(
        algorithm,
        hyperparameters,
        cv=5,
        n_iter=50,
        n_jobs=-1,
        scoring='f1_macro',
        return_train_score=True
    )
    bayes_search.fit(X_train, y_train)

    best_model = bayes_search.best_estimator_
    cv_results = bayes_search.cv_results_

    return best_model, cv_results

def get_hyperparameters(algorithm):
    return hyperparameters[algorithm.__class__.__name__]
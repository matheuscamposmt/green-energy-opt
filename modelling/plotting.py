import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def actual_vs_forecasted_df(X_test, y_test, y_pred):
    """Create a dataframe with the actual and forecasted values by year
    
    Args:
        X_test (pd.DataFrame): Test features
        y_test (pd.Series): Test target
        y_pred (pd.Series): Forecasted values
    
    Returns:
        pd.DataFrame: Dataframe with the actual and forecasted values by year
    """
    return pd.DataFrame({'actual_biomass': y_test, 'forecasted_biomass': y_pred})

def plot_forecast(model_name, X_test, y_test, y_pred, metrics=None):
    """Plot the predictions and evaluate the model using the RMSE metric
    
    Args:
        model (sklearn model): Trained model
        y_pred (pd.Series): Forecasted values
        y_test (pd.Series): Test target
        metrics (dict): Dictionary of metrics to evaluate the model
    """

    plt.figure(figsize=(15, 6))
    # add year column to y_test
    df = actual_vs_forecasted_df(X_test, y_test, y_pred)
    df['year'] = X_test['year']

    grouped = df.groupby('year').mean().reset_index()

    sns.lineplot(data=grouped, x='year', y='actual_biomass', label='Actual biomass', markers=True)
    sns.lineplot(data=grouped, x='year', y='forecasted_biomass', label='Forecasted biomass', markers=True)

    plt.title(f"[{model_name}] Forecast")
    plt.xlabel('Observation')
    plt.ylabel('Biomass (kg)')

    if metrics is not None:
        for metric, value in metrics.items():
            plt.text(0.5, 0.8 - (list(metrics.keys()).index(metric) * 0.05), f'{metric}: {value:.4f}', ha='center', va='center', transform=plt.gca().transAxes, bbox=dict(facecolor='white', alpha=0.5))

    plt.legend()
    plt.show()

def plot_actual_vs_predicted(model_name, X_test, y_test, y_pred):
    """Plot the predictions and evaluate the model using the RMSE metric
    
    Args:
        model (sklearn model): Trained model
        y_pred (pd.Series): Forecasted values
        y_test (pd.Series): Test target
    """

    df = actual_vs_forecasted_df(X_test, y_test, y_pred)

    # Plot the predictions vs the actual values
    plt.figure(figsize=(12, 6))

    sns.regplot(data=df, x='actual_biomass', y='forecasted_biomass', ci=None, line_kws={'color': 'red'})
    sns.lineplot(data=df, x='actual_biomass', y='actual_biomass', color='red')

    plt.title(f"[{model_name}] CV Actual vs Forecast")
    plt.xlabel('Actual biomass (kg)')
    plt.ylabel('Forecasted biomass (kg)')

    plt.show()


def plot_feature_importance(model, X_train):
    """Plot the feature importance of a model
    
    Args:
        model (sklearn model): Trained model
        X_train (pd.DataFrame): Train features
    """

    feature_importance = pd.DataFrame({'feature': X_train.columns, 'importance': model.feature_importances_})
    feature_importance = feature_importance.sort_values('importance', ascending=False)

    plt.figure(figsize=(12, 6))
    sns.barplot(data=feature_importance, x='importance', y='feature')
    plt.title('Feature Importance')
    plt.show()
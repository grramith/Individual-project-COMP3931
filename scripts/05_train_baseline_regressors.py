import numpy as np
import os
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error

def train_baseline_regressors():
    X_train = np.load("data/modeling/X_train.npy")
    X_val = np.load("data/modeling/X_val.npy")
    X_test = np.load("data/modeling/X_test.npy")

    y_train = np.load("data/modeling/y_train_returns.npy")
    y_val = np.load("data/modeling/y_val_returns.npy")
    y_test = np.load("data/modeling/y_test_returns.npy")

    param_grids = {
        "Linear_Regression": {
            "model_class": LinearRegression,
            "params": [{}]
        },
        "Ridge_Regression": {
            "model_class": Ridge,
            "params": [
                {"alpha": 0.01},
                {"alpha": 0.1},
                {"alpha": 1.0},
                {"alpha": 10.0},
            ]
        },
        "Random_Forest": {
            "model_class": RandomForestRegressor,
            "params": [
                {"n_estimators": 100, "max_depth": 5, "random_state": 42, "n_jobs": -1},
                {"n_estimators": 200, "max_depth": 8, "random_state": 42, "n_jobs": -1},
            ]
        },
        "Gradient_Boosting": {
            "model_class": GradientBoostingRegressor,
            "params": [
                {"n_estimators": 100, "learning_rate": 0.05, "max_depth": 3, "random_state": 42},
                {"n_estimators": 200, "learning_rate": 0.03, "max_depth": 3, "random_state": 42},
            ]
        }
    }

    best_configs = {}

    for model_name, config in param_grids.items():
        best_mae = float("inf")
        best_params = None

        for params in config["params"]:
            model = config["model_class"](**params)
            model.fit(X_train, y_train)

            val_preds = model.predict(X_val)
            val_mae = mean_absolute_error(y_val, val_preds)

            if val_mae < best_mae:
                best_mae = val_mae
                best_params = params

        best_configs[model_name] = {
            "best_val_mae": best_mae,
            "best_params": best_params
        }

    print("Validation tuning complete")
    print(best_configs)

if __name__ == "__main__":
    train_baseline_regressors()
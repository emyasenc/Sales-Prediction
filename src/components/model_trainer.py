import os
import sys
import time
from dataclasses import dataclass
import numpy as np
from sklearn.linear_model import (
    LinearRegression, Ridge, Lasso, ElasticNet, BayesianRidge
)
from sklearn.ensemble import (
    RandomForestRegressor, GradientBoostingRegressor, StackingRegressor
)
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, make_scorer
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

@dataclass
class ModelTrainerConfig:
    trained_model_file_path: str = os.path.join("artifact", "best_model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Splitting training and test input data")
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1]
            )
            
            # Define models and pipelines
            models = {
                'Linear Regression': Pipeline([
                    ('model', LinearRegression())
                ]),
                'Ridge Regression': Pipeline([
                    ('model', Ridge())
                ]),
                'Lasso Regression': Pipeline([
                    ('model', Lasso())
                ]),
                'Random Forest Regressor': Pipeline([
                    ('model', RandomForestRegressor())
                ]),
                'Support Vector Regressor': Pipeline([
                    ('model', SVR())
                ]),
                'Gradient Boosting Regressor': Pipeline([
                    ('model', GradientBoostingRegressor())
                ]),
                'XGBoost': Pipeline([
                    ('model', XGBRegressor())
                ]),
                'LightGBM': Pipeline([
                    ('model', LGBMRegressor())
                ]),
                'CatBoost': Pipeline([
                    ('model', CatBoostRegressor(silent=True))
                ]),
                'Bayesian Ridge': Pipeline([
                    ('model', BayesianRidge())
                ]),
                'Elastic Net': Pipeline([
                    ('model', ElasticNet())
                ]),
                'K-Nearest Neighbors': Pipeline([
                    ('model', KNeighborsRegressor())
                ]),
                'Stacking Regressor': Pipeline([
                    ('model', StackingRegressor(
                        estimators=[
                            ('lr', LinearRegression()),
                            ('rf', RandomForestRegressor()),
                            ('svr', SVR())
                        ],
                        final_estimator=LinearRegression()
                    ))
                ])
            }

            # Enhanced hyperparameters for grid search
            params = {
                'Random Forest Regressor': {
                    'model__n_estimators': [50, 100, 200, 500, 1000],
                    'model__max_depth': [None, 10, 20, 30],
                    'model__min_samples_split': [2, 5, 10],
                    'model__min_samples_leaf': [1, 2, 4],
                    'model__bootstrap': [True, False],
                    'model__max_features': ['log2', 'sqrt', None, 0.5, 0.8]
                },
                'Support Vector Regressor': {
                    'model__C': [0.01, 0.1, 1, 10, 100],
                    'model__epsilon': [0.01, 0.1, 0.2, 0.5],
                    'model__kernel': ['linear', 'rbf', 'poly']
                },
                'Gradient Boosting Regressor': {
                    'model__n_estimators': [100, 200, 300, 500],
                    'model__learning_rate': [0.001, 0.01, 0.1, 0.2],
                    'model__max_depth': [3, 5, 7],
                    'model__subsample': [0.7, 0.8, 0.9, 1.0]
                },
                'XGBoost': {
                    'model__n_estimators': [100, 300, 500],
                    'model__learning_rate': [0.001, 0.01, 0.1, 0.2],
                    'model__max_depth': [3, 5, 7],
                    'model__subsample': [0.7, 0.8, 0.9, 1.0],
                    'model__colsample_bytree': [0.7, 0.8, 0.9, 1.0],
                    'model__gamma': [0, 0.1, 0.2]
                },
                'LightGBM': {
                    'model__n_estimators': [100, 300, 500],
                    'model__learning_rate': [0.001, 0.01, 0.1, 0.2],
                    'model__max_depth': [-1, 3, 5, 7],
                    'model__subsample': [0.7, 0.8, 0.9, 1.0],
                    'model__colsample_bytree': [0.7, 0.8, 0.9, 1.0],
                },
                'CatBoost': {
                    'model__iterations': [100, 300, 500],
                    'model__learning_rate': [0.001, 0.01, 0.1, 0.2],
                    'model__depth': [3, 5, 7],
                    'model__l2_leaf_reg': [1, 3, 5]
                },
                'Elastic Net': {
                    'model__alpha': [0.01, 0.1, 1.0, 10.0],
                    'model__l1_ratio': [0.1, 0.5, 0.7, 0.9]
                },
                'K-Nearest Neighbors': {
                    'model__n_neighbors': [3, 5, 7, 9],
                    'model__weights': ['uniform', 'distance'],
                    'model__algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']
                }
            }

            model_performance = {}
            best_rmse = float('inf')
            best_model = None
            best_model_name = ""

            # Define scoring function
            scorer = make_scorer(mean_squared_error, greater_is_better=False, squared=False)

            # Use TimeSeriesSplit instead of k-fold cross-validation
            tscv = TimeSeriesSplit(n_splits=5)

            # Train and evaluate models
            for model_name, pipeline in models.items():
                logging.info(f"Training {model_name}")
                start_time = time.time()

                if model_name in params:
                    grid_search = RandomizedSearchCV(pipeline, params[model_name], n_iter=10, scoring=scorer, n_jobs=-1, cv=tscv, verbose=1)
                    grid_search.fit(X_train, y_train)
                    best_pipeline = grid_search.best_estimator_
                    best_score = -grid_search.best_score_
                else:
                    cv_scores = cross_val_score(pipeline, X_train, y_train, cv=tscv, scoring=scorer)
                    best_score = -cv_scores.mean()
                    pipeline.fit(X_train, y_train)
                    best_pipeline = pipeline

                elapsed_time = time.time() - start_time
                y_test_pred = best_pipeline.predict(X_test)
                r2 = r2_score(y_test, y_test_pred)

                logging.info(f"{model_name}: RMSE = {best_score:.4f}, R^2 = {r2:.4f}, Time = {elapsed_time:.2f}s")

                model_performance[model_name] = {
                    'RMSE': best_score,
                    'R^2': r2,
                    'Training Time': elapsed_time
                }

                if best_score < best_rmse:
                    best_rmse = best_score
                    best_model = best_pipeline
                    best_model_name = model_name

            logging.info(f"Best model: {best_model_name} with RMSE = {best_rmse:.4f}")

            # Save the best model
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            return model_performance

        except Exception as e:
            raise CustomException(e, sys)
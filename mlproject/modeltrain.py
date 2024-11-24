import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, r2_score
from sklearn.datasets import fetch_california_housing
import pickle
import os
import dotenv
import logging

logging.basicConfig(level=logging.INFO)

dotenv.load_dotenv()

# Set up MLflow experiment
EXPERIMENT_NAME = os.getenv("EXPERIMENT_NAME")
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))

class CaliforniaHousingTrainer:
    def __init__(self, experiment_name=EXPERIMENT_NAME):
        self.experiment_name = experiment_name
        self.setup_mlflow()

    def setup_mlflow(self):
        """Set up MLflow experiment"""
        experiment = mlflow.get_experiment_by_name(self.experiment_name)
        if experiment is None:
            mlflow.create_experiment(self.experiment_name)
        mlflow.set_experiment(self.experiment_name)

    def load_data(self):
        """Load and prepare the California housing dataset"""
        # Load data
        housing = fetch_california_housing()
        df = pd.DataFrame(housing.data, columns=housing.feature_names)
        df[housing.target_names[0]] = housing.target

        # Basic data cleaning
        # Remove outliers using IQR method
        Q1 = df.quantile(0.25)
        Q3 = df.quantile(0.75)
        IQR = Q3 - Q1
        df = df[~((df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))).any(axis=1)]

        return df, "MedHouseVal"

    def split_data(self, df, target_column, test_size=0.2, random_state=42):
        """Split data into training and testing sets and scale features"""
        X = df.drop(columns=[target_column])
        y = df[target_column]

        # Separate geographical coordinates from other features
        geo_features = ['Latitude', 'Longitude']
        features_to_scale = [col for col in X.columns if col not in geo_features]

        # Scale only non-geographical features
        scaler = StandardScaler()
        X_scaled = X.copy()
        X_scaled[features_to_scale] = scaler.fit_transform(X[features_to_scale])

        # Save the scaler
        os.makedirs('models', exist_ok=True)
        with open('models/scaler.pkl', 'wb') as f:
            pickle.dump(scaler, f)

        return train_test_split(X_scaled, y, test_size=test_size, random_state=random_state)

    def train_model(self, X_train, y_train):
        """Train model using GridSearchCV to find best hyperparameters"""
        # Define parameter grid
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [10, 20, 30, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['auto', 'sqrt']
        }

        # Initialize base model
        rf_model = RandomForestRegressor(random_state=42, verbose=1)

        # Initialize GridSearchCV
        grid_search = GridSearchCV(
            estimator=rf_model,
            param_grid=param_grid,
            cv=5,
            n_jobs=-1,
            scoring='neg_mean_squared_error',
            verbose=1,
            return_train_score=True  # This will log training scores as well
        )

        # Fit GridSearchCV
        grid_search.fit(X_train, y_train)

        # Log all tried parameters and their scores
        all_scores = []
        for mean_score, params in zip(grid_search.cv_results_['mean_test_score'],
                                    grid_search.cv_results_['params']):
            score_dict = {
                'mean_test_score': -mean_score,  # Convert back from negative MSE
                **params
            }
            all_scores.append(score_dict)

        # Save all results to a CSV for later analysis
        pd.DataFrame(all_scores).to_csv('grid_search_results.csv', index=False)

        print("\nBest parameters:", grid_search.best_params_)
        return grid_search.best_estimator_

    def evaluate_model(self, model, X_test, y_test):
        """Evaluate model performance"""
        y_pred = model.predict(X_test)

        metrics = {
            'mean_absolute_percentage_error': mean_absolute_percentage_error(y_test, y_pred),
            'mean_squared_error': mean_squared_error(y_test, y_pred),
            'root_mean_squared_error': np.sqrt(mean_squared_error(y_test, y_pred)),
            'r2_score': r2_score(y_test, y_pred)
        }

        return metrics

    def save_model(self, model, filename='best_model.pkl'):
        """Save the trained model to a file in the models directory"""
        # Create models directory if it doesn't exist
        os.makedirs('models', exist_ok=True)

        # Join path with models directory
        model_path = os.path.join('models', filename)

        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        print(f"\nModel saved as {model_path}")

    def train(self):
        """Main training pipeline"""
        with mlflow.start_run(run_name=f"rf_model_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}"):
            # Log the source code version (if using git)
            try:
                import git
                repo = git.Repo(search_parent_directories=True)
                mlflow.set_tag("commit_hash", repo.head.object.hexsha)
            except:
                logging.warning("Failed to get commit hash")

            # Load and prepare data
            logging.info("Loading and preparing data...")
            df, target_column = self.load_data()
            mlflow.log_param("data_shape", df.shape)

            # Split data
            logging.info("Splitting data...")
            X_train, X_test, y_train, y_test = self.split_data(df, target_column)
            mlflow.log_param("train_size", X_train.shape[0])
            mlflow.log_param("test_size", X_test.shape[0])

            # Train model with GridSearch
            logging.info("Training model...")
            best_model = self.train_model(X_train, y_train)

            # Log feature importances
            feature_importance = pd.DataFrame({
                'feature': X_train.columns,
                'importance': best_model.feature_importances_
            }).sort_values('importance', ascending=False)

            feature_importance.to_csv('feature_importance.csv', index=False)
            mlflow.log_artifact('feature_importance.csv')
            mlflow.log_artifact('grid_search_results.csv')

            # Evaluate model
            logging.info("Evaluating model...")
            metrics = self.evaluate_model(best_model, X_test, y_test)

            # Log parameters and metrics with MLflow
            mlflow.log_params(best_model.get_params())
            for metric_name, metric_value in metrics.items():
                mlflow.log_metric(metric_name, metric_value)

            # Print results
            logging.info("Model Performance Metrics:")
            for metric_name, metric_value in metrics.items():
                logging.info(f"{metric_name}: {metric_value}")

            # Save the model
            self.save_model(best_model)

            # Log model with MLflow
            mlflow.sklearn.log_model(best_model, "random_forest_model")

            return best_model, metrics
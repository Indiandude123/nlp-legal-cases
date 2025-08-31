import os
import logging
import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier
import yaml


log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)

logger = logging.getLogger("model_training")
logger.setLevel("DEBUG")

console_handler = logging.StreamHandler()
console_handler.setLevel("DEBUG")

log_file_path = os.path.join(log_dir, "model_training.log")
file_handler = logging.FileHandler(log_file_path)

formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)


def load_params(param_path: str) -> dict:
    """Load parameters from a YAML file"""
    try: 
        with open(param_path, "r") as f:
            params = yaml.safe_load(f)
        logger.debug("Parameters retrieved from %s", param_path)
        return params
    except FileNotFoundError:
        logger.error("File not found: %s", param_path)
        raise
    except yaml.YAMLError as e:
        logger.error("YAML Error: %s", e)
        raise
    except Exception as e:
        logger.error("Unexpected error: %s", e)
        raise


def load_data(file_path: str) -> pd.DataFrame:
    """Load data from a csv file"""
    try:
        df = pd.read_csv(file_path)
        logger.debug("Dataframe loaded from %s with shape %s successfully", file_path, df.shape)
        return df
    except pd.errors.ParserError as e:
        logger.error("Failed to parse the csv: %s", e)
        raise
    except FileNotFoundError as e:
        logger.error("File not found: %s", e)
        raise
    except Exception as e:
        logger.error("Unexpected error occurred while loading the data: %s", e)
        raise
    
def train_model(X_train: np.ndarray, y_train: np.ndarray, params: dict) -> RandomForestClassifier:
    """Trains the random forest classifier"""
    try:
        if X_train.shape[0] != y_train.shape[0]:
            raise ValueError("The number of samples in X_train and y_train must be the same.")
        
        logger.debug("Initializing Random Forest classifier with params: %s", params)
        clf = RandomForestClassifier(
            n_estimators=params['n_estimators'],
            random_state=params['random_state']
        )
        
        logger.debug("Model training started with %s samples", X_train.shape[0])
        clf.fit(X_train, y_train)
        logger.debug("Model training completed")
        
        return clf
    except ValueError as e:
        logger.error("ValueError in model training: %s", e)
        raise
    except Exception as e:
        logger.error("Error during model training: %s", e)
        raise
    
def save_model(model, file_path: str) -> None:
    """Save the model trained"""
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        with open(file_path, 'wb') as f:
            pickle.dump(model, f)
        logger.debug("Model saved to %s", file_path)
    except FileNotFoundError as e:
        logger.error("File path not found: %s", e)
        raise
    except Exception as e:
        logger.error("Error occurred in saving the model: %s", e)
        raise


def main():
    try:
        params = load_params(param_path="params.yaml")["model_training"]
        train_data = load_data("./data/processed/train_tfidf.csv")
        X_train = train_data.iloc[:, :-1].values
        y_train = train_data.iloc[:, -1].values
        
        clf = train_model(X_train, y_train, params)
        
        model_save_path = 'models/model.pkl'
        save_model(clf, model_save_path)
    
    except Exception as e:
        logger.error("Faile to complete the model building process: %s", e)
        print(f"Error: {e}")
        
        
if __name__ == "__main__":
    main()
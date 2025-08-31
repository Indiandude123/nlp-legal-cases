import os
import pandas as pd
import numpy as np
import pickle
import logging
import json
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
import yaml
from dvclive import Live

log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)

logger = logging.getLogger("model_evaluation")
logger.setLevel("DEBUG")

console_handler = logging.StreamHandler()
console_handler.setLevel("DEBUG")

log_file_path = os.path.join(log_dir, "model_evaluation.log")
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel("DEBUG")

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
    """Load data from test csv"""
    try:
        df = pd.read_csv(file_path)
        logger.debug("Data loaded from: %s", file_path)
        return df
    except pd.errors.ParserError as e:
        logger.error("Failed to parse the csv file: %s", e)
        raise
    except Exception as e:
        logger.error("Unexpected error occurred while loading the data: %s", e)
        raise

def load_model(file_path: str):
    """Load the trained model"""
    try:
        with open(file_path, "rb") as f:
            model = pickle.load(f)
        logger.debug("Model loaded successfully from: %s", file_path)
        return model
    except FileNotFoundError as e:
        logger.error("File not found: %s", e)
        raise
    except Exception as e:
        logger.error("Unexpected error occurred while loading the model: %s", e)
        raise

def evaluate_model(clf, X_test: np.ndarray, y_test: np.ndarray) -> dict:
    """Evaluate the model and return the evaluation metrics"""
    try:
        y_pred = clf.predict(X_test)
        y_pred_proba = clf.predict_proba(X_test)[:, 1]
        
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_pred_proba)
        
        eval_metrics = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "auc": auc
        }
        logger.debug("Model evaluation metrics calculated successfully")
        return eval_metrics
    except Exception as e:
        logger.error("Error during model evaluation: %s", e)
        raise
       
def save_metrics(eval_metrics: dict, file_path: str) -> None:
    """Save the evaluation metrics as a JSON file"""
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        with open(file_path, "w") as f:
            json.dump(eval_metrics, f, indent=4)
        logger.debug("Evaluation metrics saved to %s", file_path)

    except Exception as e:
        logger.error("Unexpected error occurred during saving evaluation metrics: %s", e)
        raise

def main():
    try:
        params = load_params(param_path="params.yaml")
        test_df = load_data(file_path="./data/processed/test_tfidf.csv")
        clf = load_model(file_path= "./models/model.pkl")
        
        X_test = test_df.iloc[:, :-1].values
        y_test = test_df.iloc[:, -1].values
        
        eval_metrics = evaluate_model(clf, X_test, y_test)
        
        save_metrics(eval_metrics=eval_metrics, file_path="reports/metrics.json")
    
        # Experiment tracking using dvclive
        with Live(save_dvc_exp=True) as live:
            y_pred = clf.predict(X_test)
            live.log_metric("accuracy", accuracy_score(y_test, y_pred))
            live.log_metric("precision", precision_score(y_test, y_pred))
            live.log_metric("recall", recall_score(y_test, y_pred))

            live.log_params(params)
    
    except Exception as e:
        logger.error("Failed to complete the model evaluation process: %s", e)
        print(f"Error: {e}")
    


if __name__ == "__main__":
    main()
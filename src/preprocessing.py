# This file is used for preprocessing the data

import os
import logging
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from nltk.stem.porter import PorterStemmer
import string
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
nltk.download('punkt') 

log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)

logger = logging.getLogger("preprocessing")
logger.setLevel("DEBUG")

console_handler = logging.StreamHandler()
console_handler.setLevel("DEBUG")

log_file_path = os.path.join(log_dir, "preprocessing.log")
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel("DEBUG")

formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

def transform_text(text: str) -> str:
    """Transforms the input text by converting it to lowercase, tokenizing, removing stopwords and punctuation, and stemming"""
    ps = PorterStemmer()
    text = text.lower()
    text = nltk.word_tokenize(text)
    text = [word for word in text if word.isalnum()]
    text = [word for word in text if word not in stopwords.words('english') and word not in string.punctuation]
    text = [ps.stem(word) for word in text]
    return " ".join(text)


def preprocess_df(df: pd.DataFrame, text_column: str, target_column: str) -> pd.DataFrame:
    """Preprocess the DataFrame by encoding the target column, removing duplicates and transforming the text column"""
    try:
        logger.debug("Starting the preprocessing of DataFrame")
        encoder = LabelEncoder()
        df[target_column] = encoder.fit_transform(df[target_column])
        logger.debug("Target column encoded")
        
        df = df.drop_duplicates(keep = 'first')
        logger.debug("Removed Duplicates")
        
        df.loc[:, text_column] = df[text_column].apply(transform_text)
        logger.debug("Text column transformed")
        return df
    except KeyError as e:
        logger.error("Column not found %s", e)
        raise
    
    except Exception as e:
        logger.error("Unexpected error occurred during transformation %s", e)
        raise
    

def main(text_column="text", target_column="target"):
    """Main function to load raw data, preprocess it and then save the processed data"""
    try:
        #  Fetch the train and test data
        train_data = pd.read_csv("./data/raw/train.csv")
        test_data = pd.read_csv("./data/raw/test.csv")
        logger.debug("Train and Test data loaded successfully")
        
        # Preprocess the train and test data
        train_preprocessed = preprocess_df(train_data, text_column, target_column)
        test_preprocessed = preprocess_df(test_data, text_column, target_column)
        logger.debug("Train and Test data preprocessed successfully")
        
        # Save the preprocessed train and test data
        data_path = os.path.join("./data", "interim")
        os.makedirs(data_path, exist_ok=True)
        
        train_preprocessed.to_csv(os.path.join(data_path, "train_preprocessed.csv"), index=False)
        test_preprocessed.to_csv(os.path.join(data_path, "test_preprocessed.csv"), index=False)
        
        logger.debug("Preprocessed train and test data saved to %s", data_path)
    
    except FileNotFoundError as e:
        logger.error("File Not Found: %s", e)
    except pd.errors.EmptyDataError as e:
        logger.error("No data: %s", e)
    except Exception as e:
        logger.error("Failed to complete the data preprocessing process: %s", e)
        print(f"Error: {e}")
        
if __name__ == '__main__':
    main()
import sys
import logging
import joblib
import numpy as np
import tensorflow as tf
import sys
sys.path.append('D:/MLflow-test/kidney-disease-classification')  # Adjust this path as needed

from tensorflow import keras
from tensorflow.keras.utils import to_categorical
from yaml import safe_load
from pathlib import Path
from src.logger import CustomLogger,create_log_path
from src.data.make_dataset import load_interium_X_y



NUM_CLASSES = 4


log_file_path = create_log_path("data_preprocessing")
preprocess_logger = CustomLogger(logger_name="make_dataset",log_filename=log_file_path)
preprocess_logger.set_log_level(level=logging.INFO)


def feature_scaling(X,X_name:str):
    X = X/255.0
    preprocess_logger.save_logs(msg=f'{X_name} is successfully scaled and its pixel value ranges from 0 to 1',log_level="info")
    return X


def target_ohe(y,num_classes:int,y_name:str):
    y = to_categorical(y,num_classes)
    preprocess_logger.save_logs(msg=f'{y_name} is one hot encoded to {num_classes} present and now shape of {y_name} is {y.shape}')
    return y


def save_preprocessed_data(X,y,processed_folder_path:Path,X_name:str,y_name:str):
    processed_folder = Path(processed_folder_path)
    processed_folder.mkdir(parents=True,exist_ok=True)
    X_path = processed_folder/f'{X_name}.joblib'
    y_path = processed_folder/f'{y_name}.joblib'
    joblib.dump(X,X_path)
    joblib.dump(y,y_path)
    preprocess_logger.save_logs(msg=f'{X_name} is succusfully saved in {X_path}',log_level="info")
    preprocess_logger.save_logs(msg=f'{y_name} is succusfully saved in {y_path}',log_level="info")


def load_preprocessed_data(processed_folder_path:Path,X_name:str,y_name:str):
    processed_folder = Path(processed_folder_path)
    X_path = processed_folder/f'{X_name}.joblib'
    y_path = processed_folder/f'{y_name}.joblib'
    X = joblib.load(X_path)
    y = joblib.load(y_path)
    preprocess_logger.save_logs(msg=f'{X_name} is succusfully loaded from {X_path}',log_level="info")
    preprocess_logger.save_logs(msg=f'{y_name} is succusfully loaded from {y_path}',log_level="info")
    return X,y


def main():
    X_train_filename = sys.argv[1]
    y_train_filename = sys.argv[2]
    X_val_filename = sys.argv[3]
    y_val_filename = sys.argv[4]

    current_path = Path(__file__)
    root_path = current_path.parent.parent.parent

    INTERIUM_FOLDER_PATH = root_path/'data'/'interium'
    PROCESSED_FOLDER_PATH = root_path/'data'/'processed'
    NUM_CLASSES = 4

    X_train,y_train = load_interium_X_y(interium_folder_path=INTERIUM_FOLDER_PATH,X_name=X_train_filename,y_name=y_train_filename)
    X_train_preprocessed = feature_scaling(X_train,"X_train_preprocessed")
    y_train_preprocessed = target_ohe(y_train,num_classes=NUM_CLASSES,y_name="y_train_preprocessed")
    save_preprocessed_data(X=X_train_preprocessed,y=y_train_preprocessed,processed_folder_path = PROCESSED_FOLDER_PATH,X_name='X_train_preprocessed',y_name='y_train_preprocessed')

    X_val,y_val = load_interium_X_y(interium_folder_path=INTERIUM_FOLDER_PATH,X_name=X_val_filename,y_name=y_val_filename)
    X_val_preprocessed = feature_scaling(X_val,"X_val_preprocessed")
    y_val_preprocessed = target_ohe(y_val,num_classes=NUM_CLASSES,y_name="y_val_preprocessed")
    save_preprocessed_data(X=X_val_preprocessed,y=y_val_preprocessed,processed_folder_path = PROCESSED_FOLDER_PATH,X_name='X_val_preprocessed',y_name='y_val_preprocessed')


if __name__ == "__main__":
    main()



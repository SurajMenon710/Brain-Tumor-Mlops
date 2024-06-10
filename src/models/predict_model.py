import logging
import sys
import cv2
import datetime as dt
import sys
sys.path.append('D:/MLflow-test/kidney-disease-classification')  # Adjust this path as needed

import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import Sequential
from pathlib import Path
from yaml import safe_load
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import load_model
from keras.layers import Dense,Conv2D,MaxPooling2D,Flatten,Dropout
from sklearn.metrics import classification_report
from src.logger import CustomLogger,create_log_path
from src.features.data_preprocessing import load_preprocessed_data




log_file_path= create_log_path("predict_model")
predict_model_logger = CustomLogger(logger_name="predict_model",log_filename=log_file_path)
predict_model_logger.set_log_level(level=logging.INFO)


CLASSES = ['glioma_tumor','meningioma_tumor','no_tumor','pituitary_tumor']
model_name ='CNNmodel.h5'

def load_raw_image_data(input_dir_path : Path,classes : list,image_size:list) -> list:
    data = list()
    input_dir_path = Path(input_dir_path)
    for category in classes:
        path = input_dir_path/category
        class_num = classes.index(category)
        count = 0
        for image_path in path.iterdir():
            if image_path.is_file():
                img_array = cv2.imread(str(image_path),cv2.IMREAD_GRAYSCALE)
                resized_array = cv2.resize(img_array,(image_size[0],image_size[1]))
                data.append([resized_array,class_num])
                count += 1
        predict_model_logger.save_logs(msg = f"Loaded {count} test images out of {len([item for item in path.iterdir() if item.is_file()])} test images of Class {category}",log_level="info")
    return data


def create_X_y(data:list,img_size:list):
    X = [image for image,label in data]
    y = [label for image,label in data]
    X = np.array(X).reshape(-1,img_size[0],img_size[1],img_size[2])
    predict_model_logger.save_logs(msg = f"Shape of X test is {X.shape} where the {X.shape[0]} represents no. of samples ,[{X.shape[1]},{X.shape[2]}] represents height and width of image and channel is {X.shape[3]}",log_level="info")
    return X,y

def feature_scaling(X,X_name:str):
    X = X/255.0
    predict_model_logger.save_logs(msg=f'{X_name} is successfully scaled and its pixel value ranges from 0 to 1',log_level="info")
    return X


def target_ohe(y,num_classes:int,y_name:str):
    y = to_categorical(y,num_classes)
    predict_model_logger.save_logs(msg=f'{y_name} is one hot encoded to {num_classes} present and now shape of {y_name} is {y.shape}')
    return y


def predict_category(model,X):
    y_pred = model.predict(X)
    return y_pred


def save_classification_report(report_save_path:Path,report):
    report_save_path = Path(report_save_path)
    report_save_path.mkdir(parents=True, exist_ok=True)
    current_date_str = dt.date.today().strftime("%d-%m-%Y_%H-%M-%S")
    report_file_path = report_save_path/f'{current_date_str}_classification_report.txt'
    with open(report_file_path, 'w') as file:
        file.write(report)
    predict_model_logger.save_logs(msg=f'Classification report saved at {report_file_path}')



def main():
    CLASSES = ['glioma_tumor','meningioma_tumor','no_tumor','pituitary_tumor']
    IMAGE_SIZE = [200,200,1]
    NUM_CLASSES = 4

    input_file_name = sys.argv[1]
    current_path = Path(__file__)
    root_path = current_path.parent.parent.parent
    test_data_path = root_path/input_file_name

    data = load_raw_image_data(input_dir_path=test_data_path,classes=CLASSES,image_size=IMAGE_SIZE)
    X_test,y_test = create_X_y(data=data,img_size=IMAGE_SIZE)
    X_test = feature_scaling(X=X_test,X_name="X_test")
    y_test = target_ohe(y=y_test,num_classes=NUM_CLASSES,y_name="y_test")

    model_name ='CNNmodel.h5'
    model_path = root_path/'models'/model_name
    model = load_model(model_path)
    y_pred = predict_category(model=model,X=X_test)

    y_pred = np.argmax(y_pred,axis=1)
    y_test = np.argmax(y_test,axis=1)

    report = classification_report(y_true=y_test,y_pred=y_pred,target_names=CLASSES)
    print(report)

    save_report_path = root_path/'reports'
    save_report = save_classification_report(report_save_path=save_report_path,report=report)


if __name__ == "__main__":
    main()
















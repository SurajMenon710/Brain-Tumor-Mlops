import logging
import sys
import sys
sys.path.append('D:/MLflow-test/brain-disease-classification')  # Adjust this path as needed

import cv2
import joblib
import numpy as np
from yaml import safe_load
from src.logger import CustomLogger,create_log_path
from sklearn.model_selection import train_test_split
from pathlib import Path




log_file_path = create_log_path("make_dataset")
dataset_logger = CustomLogger(logger_name="make_dataset",log_filename=log_file_path)
dataset_logger.set_log_level(level=logging.INFO)

CLASSES = ['glioma_tumor','meningioma_tumor','no_tumor','pituitary_tumor']

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
        dataset_logger.save_logs(msg = f"Loaded {count} images out of {len([item for item in path.iterdir() if item.is_file()])} images of Class {category}",log_level="info")
    return data


def create_X_y(data:list,img_size:list):
    X = [image for image,label in data]
    y = [label for image,label in data]
    X = np.array(X).reshape(-1,img_size[0],img_size[1],img_size[2])
    dataset_logger.save_logs(msg = f"Shape of X is {X.shape} where the {X.shape[0]} represents no. of samples ,[{X.shape[1]},{X.shape[2]}] represents height and width of image and channel is {X.shape[3]}",log_level="info")
    return X,y


def train_val_split(test_size:float,X:list,y:list,random_state:int):
    X_train,X_val,y_train,y_val = train_test_split(X,y,test_size=test_size,random_state=random_state)
    dataset_logger.save_logs(msg=f'Data is split into train split with shape {X_train.shape} and val split with shape {X_val.shape}',log_level='info')
    dataset_logger.save_logs(msg=f'The parameter values are {test_size} for test_size and {random_state} for random_state',log_level='info')
    return X_train,X_val,y_train,y_val


def save_interium_data(X,y,interium_folder_path:Path,X_name:str,y_name:str):
    interium_folder = Path(interium_folder_path)
    interium_folder.mkdir(parents=True,exist_ok=True)
    X_path = interium_folder/f'{X_name}.joblib'
    y_path = interium_folder/f'{y_name}.joblib'
    joblib.dump(X,X_path)
    joblib.dump(y,y_path)
    dataset_logger.save_logs(msg=f'{X_name} is succusfully saved in {X_path}',log_level="info")
    dataset_logger.save_logs(msg=f'{y_name} is succusfully saved in {y_path}',log_level="info")


def load_interium_X_y(interium_folder_path:Path,X_name:str,y_name:str):
    interium_folder = Path(interium_folder_path)
    X_path = interium_folder/f'{X_name}.joblib'
    y_path = interium_folder/f'{y_name}.joblib'
    X = joblib.load(X_path)
    y = joblib.load(y_path)
    dataset_logger.save_logs(msg=f'{X_name} is succusfully loaded from {X_path}',log_level="info")
    dataset_logger.save_logs(msg=f'{y_name} is succusfully loaded from {y_path}',log_level="info")
    return X,y


def read_params(input_file):
    try:
        with open(input_file) as f:
            params_file = safe_load(f)
            
    except FileNotFoundError as e:
        dataset_logger.save_logs(msg='Parameters file not found, Switching to default values for train test split and image size',log_level='error')
        default_dict = {'image_size': [200,200,1],
                        'test_size': 0.25,
                        'random_state': None}
        image_size = default_dict['image_size']
        test_size = default_dict['test_size']
        random_state = default_dict['random_state']
        return image_size,test_size, random_state
    else:
        dataset_logger.save_logs(msg=f'Parameters file read successfully',log_level='info')
        image_size = params_file['make_dataset']['image_size']
        test_size = params_file['make_dataset']['test_size']
        random_state = params_file['make_dataset']['random_state']
        return image_size,test_size, random_state


def main():
    CLASSES = ['glioma_tumor','meningioma_tumor','no_tumor','pituitary_tumor']
    COLOUR_MODE = "grayscale"

    input_file_name = sys.argv[1]
    current_path = Path(__file__)
    root_path = current_path.parent.parent.parent
    raw_input_dir_path = root_path/'data'/'raw'/input_file_name

    image_size,test_size, random_state = read_params('params.yaml')

    data = load_raw_image_data(input_dir_path=raw_input_dir_path,classes=CLASSES,image_size=image_size)

    X,y = create_X_y(data=data,img_size=image_size)
    X_train,X_val,y_train,y_val = train_val_split(test_size=test_size,X=X,y=y,random_state=random_state)

    INTERIUM_FOLDER_PATH = root_path/'data'/'interium'
    INTERIUM_FOLDER_PATH.mkdir(exist_ok=True)
    save_interium_data(X=X_train,y=y_train,interium_folder_path=INTERIUM_FOLDER_PATH,X_name='X_train',y_name='y_train')
    save_interium_data(X=X_val,y=y_val,interium_folder_path=INTERIUM_FOLDER_PATH,X_name='X_val',y_name='y_val')


if __name__ == "__main__":
    main()


        




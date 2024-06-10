import logging
import sys
import cv2
import sys
sys.path.append('D:/MLflow-test/brain-disease-classification')  # Adjust this path as needed

import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import Sequential
from pathlib import Path
from yaml import safe_load
from keras.layers import Dense,Conv2D,MaxPooling2D,Flatten,Dropout
from src.logger import CustomLogger,create_log_path
from src.features.data_preprocessing import load_preprocessed_data



log_file_path = create_log_path("train_model")
train_model_logger = CustomLogger(logger_name="train_model",log_filename=log_file_path)
train_model_logger.set_log_level(level=logging.INFO)


def tf_build_model(image_size:list):
    model = Sequential()
    model.add(Conv2D(32,kernel_size=(3,3),strides=1,padding="same",activation="relu",input_shape = (image_size[0],image_size[1],image_size[2])))
    model.add(MaxPooling2D(pool_size=(2,2),padding = "valid"))
    model.add(Dropout(0.3))
    model.add(Conv2D(64,kernel_size=(3,3),strides=1,padding="same",activation="relu"))
    model.add(MaxPooling2D(pool_size=(2,2),padding = "valid"))
    model.add(Dropout(0.3))
    model.add(Flatten())
    model.add(Dropout(0.5))
    model.add(Dense(512,activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4,activation = 'softmax'))

    total_params = model.count_params()
    trainable_params = np.sum([np.prod(v.shape) for v in model.trainable_weights])
    non_trainable_params = np.sum([np.prod(v.shape) for v in model.non_trainable_weights])
    train_model_logger.save_logs(msg=f'Model has total parameters : {total_params} out of which {trainable_params} are trainable parameters and {non_trainable_params} are non-trainable parameters',log_level="info")
    return model


def tf_compile_model(model):
    model.compile(optimizer="adam",loss="categorical_crossentropy",metrics = ['accuracy'])
    return model


def tf_train_model(model,X_train,y_train,batch_size:int,epochs:int,validation_data:tuple):
    model.fit(X_train,y_train,batch_size=batch_size ,epochs=epochs,validation_data=validation_data,verbose=1)
    return model


def tf_save_model(model,save_model_path:Path,model_name:str):
    save_model_path = Path(save_model_path)
    model.save(save_model_path/f'{model_name}.h5')


def read_params(input_file):
    try:
        with open(input_file) as f:
            params_file = safe_load(f)
            
    except FileNotFoundError as e:
        train_model_logger.save_logs(msg='Parameters file not found, Switching to default values for epochs,batch size and model name',log_level='error')
        default_dict = {'image_size':[200,200,1],'epochs': 10,
                        'batch_size': 32,
                        'model_name': "CNN_model"}
        image_size = default_dict['image_size']
        epochs = default_dict['epochs']
        batch_size = default_dict['batch_size']
        model_name = default_dict['model_name']
        return image_size,epochs,batch_size,model_name
    else:
        train_model_logger.save_logs(msg=f'Parameters file read successfully',log_level='info')
        image_size = params_file['make_dataset']['image_size']
        epochs = params_file['train_model']['epochs']
        batch_size = params_file['train_model']['batch_size']
        model_name = params_file['train_model']['model_name']
        return image_size,epochs,batch_size,model_name


def main():
    current_path = Path(__file__)
    root_path = current_path.parent.parent.parent

    input_file_name = sys.argv[1]
    processed_data_path = root_path/input_file_name
    X_train_preprocessed,y_train_preprocessed = load_preprocessed_data(processed_folder_path=processed_data_path,X_name="X_train_preprocessed",y_name="y_train_preprocessed")
    X_val_preprocessed,y_val_preprocessed = load_preprocessed_data(processed_folder_path=processed_data_path,X_name="X_val_preprocessed",y_name="y_val_preprocessed")

    image_size,epochs,batch_size,model_name = read_params('params.yaml')
    model = tf_build_model(image_size=image_size)
    model = tf_compile_model(model)
    model = tf_train_model(model=model,X_train=X_train_preprocessed,y_train=y_train_preprocessed,batch_size=batch_size,epochs=epochs,validation_data=(X_val_preprocessed,y_val_preprocessed))

    model_output_path = root_path/'models'
    model_output_path.mkdir(exist_ok=True)
    tf_save_model(model=model,save_model_path=model_output_path,model_name=model_name)


if __name__ == "__main__":
    main()

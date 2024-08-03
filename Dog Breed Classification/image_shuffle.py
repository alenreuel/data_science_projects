import os
import random
import shutil
import pandas as pd
import numpy as np
from PIL import Image

def image_shuffler(dataset_path, train_set_size=0.7):

    data_path = dataset_path

    #create 3 new_folders
    test_size = (1-train_set_size)/2
    train_folder = os.path.join(data_path, "training/train_set")
    val_folder = os.path.join(data_path, "training/val_set")
    test_folder = os.path.join(data_path, "training/test_set")
    for i in os.listdir(data_path):
        if i not in {train_folder, val_folder, test_folder}:
            cur_dir = os.path.join(data_path, i)
            for j in os.listdir(cur_dir):
                dest_number = random.random()
                if dest_number<train_set_size:
                    if os.path.exists(os.path.join(train_folder, i)):
                        shutil.copy(os.path.join(cur_dir, j), os.path.join(train_folder, i))
                    else:
                        os.makedirs(os.path.join(train_folder, i))
                        shutil.copy(os.path.join(cur_dir, j), os.path.join(train_folder, i))
                elif (train_set_size-0.15)<dest_number<(train_set_size+0.15):
                    if os.path.exists(os.path.join(val_folder, i)):
                        shutil.copy(os.path.join(cur_dir, j), os.path.join(val_folder, i))
                    else:
                        os.makedirs(os.path.join(val_folder, i))
                        shutil.copy(os.path.join(cur_dir, j), os.path.join(val_folder, i))
                else:
                    if os.path.exists(os.path.join(test_folder, i)):
                        shutil.copy(os.path.join(cur_dir, j), os.path.join(test_folder, i))
                    else:
                        os.makedirs(os.path.join(test_folder, i))
                        shutil.copy(os.path.join(cur_dir, j), os.path.join(test_folder, i))


def create_dataframe_for_images(data_path, resize_dims = (64,64)):
    ls_values = []
    for i in os.listdir(data_path):
        if i != "training":
            cur_dir = os.path.join(data_path, i)
            for j in os.listdir(cur_dir):
                image = Image.open(os.path.join(cur_dir,j))
                image.resize(resize_dims)
                data = np.array(image.getdata()).reshape(resize_dims[0]*resize_dims[1],1)
                ls_values.append(list(data)+[i]) 

    return pd.DataFrame(ls_values, columns= [i for i in range(64*64)]+["label"] )





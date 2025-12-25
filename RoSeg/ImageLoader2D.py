import glob
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
import cv2


def load_data(img_height, img_width, images_to_be_loaded, dataset, folder_path):
    print(f"Loading dataset type: {dataset}") 
    IMAGES_PATH = folder_path + 'images/'
    MASKS_PATH = folder_path + 'masks/'

    if dataset == 'kvasir':
        train_ids = glob.glob(IMAGES_PATH + "*.jpg") + glob.glob(IMAGES_PATH + "*.png")

    elif dataset == 'cvc-clinicdb':
        train_ids = glob.glob(IMAGES_PATH + "*.tif")

    elif dataset in ['cvc-colondb', 'etis-laribpolypdb']:
        train_ids = glob.glob(IMAGES_PATH + "*.png")
    
    elif dataset == 'unseen':
        train_ids = glob.glob(IMAGES_PATH + "*.png")
        
    elif dataset == 'jpg':
        train_ids = glob.glob(IMAGES_PATH + "*.jpg") 
        
    else:
        raise ValueError(f"Unknown dataset type: {dataset}")

    if images_to_be_loaded == -1:
        images_to_be_loaded = len(train_ids)

    X_train = torch.zeros((images_to_be_loaded, 3, img_height, img_width), dtype=torch.float32)
    Y_train = torch.zeros((images_to_be_loaded, 1, img_height, img_width), dtype=torch.float32)

    print('Resizing and loading training images and masks:', images_to_be_loaded)
    for n, id_ in tqdm(enumerate(train_ids), total=images_to_be_loaded):
        if n == images_to_be_loaded:
            break

        image_path = id_
        mask_path = image_path.replace("images", "masks")

        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (img_width, img_height), interpolation=cv2.INTER_CUBIC)
        image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0

        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, (img_width, img_height), interpolation=cv2.INTER_NEAREST)
        mask = torch.from_numpy(mask).float()
        mask = (mask >= 127).float()
        mask = mask.unsqueeze(0)

        X_train[n] = image
        Y_train[n] = mask

    return X_train, Y_train

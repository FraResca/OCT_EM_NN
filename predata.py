from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.utils import shuffle
import os
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.neighbors import NearestNeighbors


def corr_mat(X, Y):
    df = pd.DataFrame(np.concatenate([X, Y], axis=1))
    print('\nDataFrame originale:')
    print(df)
    print('\nMatrice di correlazione:')
    print(df.corr())

def data_prep_multi():
    df = pd.read_excel('XL.xlsx', engine='openpyxl')

    data = df.to_numpy()

    ids = data[:, 0]  # first column

    X = data[:, 1:-4]  # all columns except first and last 4
    Y = data[:, -4:]  # last 4 columns

    # print(X)
    # print(Y)

    labenc = LabelEncoder()
    X[:, 0] = labenc.fit_transform(X[:, 0])

    ohenc = OneHotEncoder()
    multiEnc = ohenc.fit_transform(X[:, 2].reshape(-1, 1)).toarray()

    X[:, 10] = X[:, 10] - 1

    X = np.delete(X, 2, axis=1)
    X = np.concatenate((X, multiEnc), axis=1)

    X = X.astype(float)

    scaler = StandardScaler()
    X_norm = scaler.fit_transform(X)
    Y_norm = scaler.fit_transform(Y)

    images = []

    for id in ids:
        if id < 10:
            image_file = os.path.join(f'OCT_crops/0{id}_0.tif')
        else:
            image_file = os.path.join(f'OCT_crops/{id}_0.tif')

        img = image.load_img(image_file, target_size=(512,512))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)

        images.append(img_array)

    images = np.concatenate(images)

    #print(X_norm)
    #print(Y_norm)

    return X_norm, images, Y_norm

def splits(X_norm, images, Y_norm):
    X_train, X_test, Y_train, Y_test = train_test_split(X_norm, Y_norm, test_size=0.2, random_state=42)
    images_train, images_test, _, _ = train_test_split(images, Y_norm, test_size=0.2, random_state=42)

    return X_train, X_test, images_train, images_test, Y_train, Y_test

def data_aug(X, images, Y):
    numerical_cols_X = [1, 5, 6]
    numerical_cols_Y = [2, 3]
    seed=42
    
    datagen = ImageDataGenerator(rotation_range=10)
    datagen.fit(images, seed=seed)

    augmented_images = []
    for img in images:
        img = img.reshape((1,) + img.shape)
        for batch in datagen.flow(img, batch_size=1, seed=seed):
            augmented_images.append(batch[0])
            break

    np.random.seed(seed)

    X_aug = X.copy()
    for col in numerical_cols_X:
        mean = X[:, col].mean()
        std = X[:, col].std()
        noise = np.random.normal(mean, std, X[:, col].shape)
        X_aug[:, col] += noise

    Y_aug = Y.copy()
    for col in numerical_cols_Y:
        mean = Y[:, col].mean()
        std = Y[:, col].std()
        noise = np.random.normal(mean, std, Y[:, col].shape)
        Y_aug[:, col] += noise

    X_combined = np.concatenate([X, X_aug])
    images_combined = np.concatenate([images, augmented_images])
    Y_combined = np.concatenate([Y, Y_aug])

    return X_combined, images_combined, Y_combined
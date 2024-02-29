from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input
from sklearn.preprocessing import LabelEncoder, StandardScaler
import os
import numpy as np
import pandas as pd

def data_prep():
    df = pd.read_excel('XLsquartabile.xlsx', engine='openpyxl')
    data = df.to_numpy()

    ids = data[:, 0]  # all rows, first column

    X = data[:, 1:-1]  # all rows, all columns except first and last
    y = data[:, -1]  # all rows, last column

    le = LabelEncoder()

    X[:, 0] = le.fit_transform(X[:, 0])
    X[:, 1] = le.fit_transform(X[:, 1])

    X = X.astype(float)

    scaler = StandardScaler()
    X_norm = scaler.fit_transform(X)

    y = y.reshape(-1, 1)
    y_norm = scaler.fit_transform(y)

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

    # Convert the list of images to a numpy array
    images = np.concatenate(images)

    return [images, X_norm], y_norm



'''
# Preprocessing dell'immagine
img = image.load_img('02_0_cropped.tif', target_size=(512,512))

#print(img.size)

img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array = preprocess_input(img_array)

#print(img_array.shape)

# Estrazione features

#features = model.predict(img_array)
'''
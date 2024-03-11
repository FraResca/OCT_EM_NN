from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy, MeanSquaredError
from tensorflow.keras.models import Model, load_model
from sklearn.utils import shuffle
from scipy.ndimage import zoom
from models import hybnet_multi
from predata import data_prep_multi, splits, data_aug
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import sys


def download_vgg16():
    model = VGG16(weights='imagenet', include_top=False, input_shape=(512, 512, 3))
    model.save('vgg16.h5')

def custom_loss_imbalance(y_true, y_pred):
    # Flatten the tensor and convert to int
    y_true_flat = tf.reshape(y_true[:,:2], [-1])
    y_true_int = tf.cast(y_true_flat, tf.int32)

    # Calculate class counts
    unique, _, class_counts = tf.unique_with_counts(y_true_int)

    # Calculate class weights
    max_count = tf.cast(tf.reduce_max(class_counts), tf.float32)
    class_weights = tf.math.divide_no_nan(max_count, tf.cast(class_counts, tf.float32))

    # Calculate binary cross-entropy loss
    binary_loss = BinaryCrossentropy()(y_true[:,:2], y_pred[:,:2])

    # Apply class weights
    weighted_binary_loss = tf.reduce_sum(binary_loss * class_weights)

    # Calculate mean squared error loss
    mse_loss = MeanSquaredError()(y_true[:,2:], y_pred[:,2:])

    return weighted_binary_loss + mse_loss

def custom_loss(y_true, y_pred):
    binary_loss = BinaryCrossentropy()(y_true[:,:2], y_pred[:,:2])
    mse_loss = MeanSquaredError()(y_true[:,2:], y_pred[:,2:])

    return binary_loss + mse_loss

def train_hybrid_multi(n_epochs, aug):
    hybrid_model = hybnet_multi()
    # hybrid_model.summary()

    X, images, Y = data_prep_multi()
    X, images, Y = shuffle(X, images, Y, random_state=42)
    if aug:
        X, images, Y = data_aug(X, images, Y)

    X_train, _, images_train, _, Y_train, _ = splits(X, images, Y)

    lr = 0.001
    hybrid_model.compile(optimizer=Adam(learning_rate=lr), loss=custom_loss, metrics=['mean_absolute_error'])
    hybrid_model.fit([images_train, X_train], Y_train, epochs=n_epochs, batch_size=1)
    hybrid_model.save('hybrid_model.h5')

def train_hybrid_multi_imbalance(n_epochs, aug):
    hybrid_model = hybnet_multi()
    # hybrid_model.summary()

    X, images, Y = data_prep_multi()
    X, images, Y = shuffle(X, images, Y, random_state=42)
    if aug:
        X, images, Y = data_aug(X, images, Y)

    X_train, _, images_train, _, Y_train, _ = splits(X, images, Y)

    lr = 0.001
    hybrid_model.compile(optimizer=Adam(learning_rate=lr), loss=custom_loss_imbalance, metrics=['mean_absolute_error'])
    hybrid_model.fit([images_train, X_train], Y_train, epochs=n_epochs, batch_size=1)
    hybrid_model.save('hybrid_model_imbalance.h5')


def eval_hybrid_multi(modelname):
    hybrid_model = load_model(modelname, custom_objects={'custom_loss': custom_loss, 'custom_loss_imbalance': custom_loss_imbalance})

    X, images, Y = data_prep_multi()
    _, X_test, _, images_test, _, Y_test = splits(X, images, Y)

    hybrid_model.evaluate([images_test, X_test], Y_test)

def visualize_attention(modelname):
    X, images, Y = data_prep_multi()
    _, X_test, _, images_test, _, _ = splits(X, images, Y)


    hybrid_model = load_model(modelname, custom_objects={'custom_loss': custom_loss, 'custom_loss_imbalance': custom_loss_imbalance})

    multiply_layer_index = -1
    for i, layer in enumerate(hybrid_model.layers):
        if 'multiply' in layer.name:
            multiply_layer_index = i
            break

    attention_model = Model(inputs=hybrid_model.input, outputs=hybrid_model.layers[multiply_layer_index].output)

    attention_map = attention_model.predict([images_test, X_test])

    attention_map -= attention_map.min()
    attention_map /= attention_map.max()

    attention_map_2d = np.sum(attention_map, axis=-1)

    attention_map_resized = np.array([zoom(img, (images_test[0].shape[0]/img.shape[0], images_test[0].shape[1]/img.shape[1])) for img in attention_map_2d])

    attention_map_resized = attention_map_resized.astype('float32') / attention_map_resized.max()

    num_images = len(images_test)
    '''
    num_cols = 2
    num_rows = num_images // num_cols

    if num_images % num_cols:
        num_rows += 1

    
    fig = plt.figure(figsize=(10, num_rows * 5))

    for i, (img, att_map) in enumerate(zip(images_test, attention_map_resized), start=1):
        ax = fig.add_subplot(num_rows, num_cols, i)
        ax.imshow(img, cmap='gray')
        ax.imshow(att_map, cmap='jet', alpha=0.5)

    plt.savefig(f'attention_{modelname}_{i}.png')
    '''
    
    for i in range(num_images):
        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        ax[0].imshow(images_test[i], cmap='gray')
        ax[1].imshow(images_test[i], cmap='gray')
        ax[1].imshow(attention_map_resized[i], cmap='jet', alpha=0.5)
        plt.savefig(f'attention_maps/attention_{modelname}_{i}.png')
    
    #plt.show()

if __name__ == '__main__':
    if len(sys.argv) != 4:
        print('Usage: python trainer.py <number of epochs> <augmented yes/no> <vis yes/no>')
        sys.exit(1)
    if sys.argv[2] == 'yes':
        aug = True
    else:
        aug = False

    # download_vgg16()
    train_hybrid_multi(int(sys.argv[1]), aug)
    eval_hybrid_multi('hybrid_model.h5')

    train_hybrid_multi_imbalance(int(sys.argv[1]), aug)
    eval_hybrid_multi('hybrid_model_imbalance.h5')
    
    if sys.argv[3] == 'yes':
        visualize_attention('hybrid_model.h5')
        visualize_attention('hybrid_model_imbalance.h5')
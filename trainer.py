from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy, MeanSquaredError
from tensorflow.keras.models import Model, load_model
from sklearn.utils import shuffle
from scipy.ndimage import zoom
from models import *
from predata import *
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import sys


def download_vgg16():
    model = VGG16(weights='imagenet', include_top=False, input_shape=(512, 512, 3))
    model_noruler = VGG16(weights='imagenet', include_top=False, input_shape=(444, 444, 3))
    model.save('vgg16.h5')
    model_noruler.save('vgg16_noruler.h5')

def custom_loss(y_true, y_pred):
    binary_loss = BinaryCrossentropy()(y_true[:,:2], y_pred[:,:2])
    mse_loss = MeanSquaredError()(y_true[:,2:], y_pred[:,2:])

    return binary_loss + mse_loss

def custom_loss_imbalance(y_true, y_pred):
    y_true_flat = tf.reshape(y_true[:,:2], [-1])
    y_true_int = tf.cast(y_true_flat, tf.int32)

    _, _, class_counts = tf.unique_with_counts(y_true_int)

    total_count = tf.reduce_sum(class_counts)
    class_weights = tf.math.divide_no_nan(tf.cast(total_count, tf.float32), tf.cast(class_counts, tf.float32))
    class_weights = tf.reshape(class_weights, (1, -1))  # reshape class_weights to have shape (1, 2)
    class_weights = tf.repeat(class_weights, tf.shape(y_true)[0], axis=0)  # repeat it to match the batch size

    binary_loss = tf.nn.weighted_cross_entropy_with_logits(y_true[:,:2], y_pred[:,:2], class_weights)

    mse_loss = MeanSquaredError()(y_true[:,2:], y_pred[:,2:])

    return binary_loss + mse_loss

def custom_loss_visus(y_true, y_pred):
    binary_loss = BinaryCrossentropy()(y_true[:,:2], y_pred[:,:2])
    mse_loss = MeanSquaredError()(y_true[:,3:], y_pred[:,3:])

    return binary_loss + mse_loss

def custom_loss_imbalance_visus(y_true, y_pred):
    y_true_flat = tf.reshape(y_true[:,:2], [-1])
    y_true_int = tf.cast(y_true_flat, tf.int32)

    _, _, class_counts = tf.unique_with_counts(y_true_int)

    total_count = tf.reduce_sum(class_counts)
    class_weights = tf.math.divide_no_nan(tf.cast(total_count, tf.float32), tf.cast(class_counts, tf.float32))
    class_weights = tf.reshape(class_weights, (1, -1))  # reshape class_weights to have shape (1, 2)
    class_weights = tf.repeat(class_weights, tf.shape(y_true)[0], axis=0)  # repeat it to match the batch size

    binary_loss = tf.nn.weighted_cross_entropy_with_logits(y_true[:,:2], y_pred[:,:2], class_weights)

    mse_loss = MeanSquaredError()(y_true[:,3:], y_pred[:,3:])

    return binary_loss + mse_loss

    
def train_general(n_epochs, balance, aug, noruler, outmodel_name):
    hybrid_model = hybnet_general(noruler)
    hybrid_model.summary()

    X, images, Y = data_prep_general(noruler, False)
    X, images, Y = shuffle(X, images, Y, random_state=42)
    if aug:
        X, images, Y = data_aug(X, images, Y)
    X_train, _, images_train, _, Y_train, _ = splits(X, images, Y)

    lr = 0.001
    if balance == True:
        hybrid_model.compile(optimizer=Adam(learning_rate=lr), loss=custom_loss_imbalance, metrics=['mean_absolute_error'])
    else:
        hybrid_model.compile(optimizer=Adam(learning_rate=lr), loss=custom_loss, metrics=['mean_absolute_error'])

    hybrid_model.fit([images_train, X_train], Y_train, epochs=n_epochs, batch_size=1)
    hybrid_model.save(f'{outmodel_name}.h5')

def eval_general(modelname, balance, noruler, vis):
    if balance == True:
        hybrid_model = load_model(f'{modelname}.h5', custom_objects={'custom_loss_imbalance': custom_loss_imbalance, 'custom_loss_imbalance_visus': custom_loss_imbalance_visus})
    else:
        hybrid_model = load_model(f'{modelname}.h5', custom_objects={'custom_loss': custom_loss, 'custom_loss_visus': custom_loss_visus})
    
    X, images, Y = data_prep_general(noruler, vis)
    X, images, Y = shuffle(X, images, Y, random_state=42)
    # X, images, Y = data_aug(X, images, Y)
    _, X_test, _, images_test, _, Y_test = splits(X, images, Y)

    hybrid_model.evaluate([images_test, X_test], Y_test)

def visualize_attention_general(modelname, noruler, balance, vis):
    X, images, Y = data_prep_general(noruler, vis)
    # X, images, Y = data_aug(X, images, Y)
    _, X_test, _, images_test, _, _ = splits(X, images, Y)

    if balance == True:
        hybrid_model = load_model(f'{modelname}.h5', custom_objects={'custom_loss_imbalance': custom_loss_imbalance, 'custom_loss_imbalance_visus': custom_loss_imbalance_visus})
    else:
        hybrid_model = load_model(f'{modelname}.h5', custom_objects={'custom_loss': custom_loss, 'custom_loss_visus': custom_loss_visus})
    
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

    for i in range(num_images):
        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        ax[0].imshow(images_test[i], cmap='gray')
        ax[1].imshow(images_test[i], cmap='gray')
        ax[1].imshow(attention_map_resized[i], cmap='jet', alpha=0.5)
        plt.savefig(f'attention_maps/attention_{modelname}_{i}.png')

def train_visus(modelname, balance):
    
    if balance == True:
        custom_object = {'custom_loss_imbalance': custom_loss_imbalance}
    else:
        custom_object = {'custom_loss': custom_loss}

    model = load_model(f'{modelname}.h5', custom_objects=custom_object)

    X, images, Y = data_prep_general(noruler, True)
    X, images, Y = shuffle(X, images, Y, random_state=42)    
    X_train, _, images_train, _, Y_train, _ = splits(X, images, Y)

    x = model.layers[-2].output

    output = Dense(5, name='output')(x)

    new_model = Model(inputs=model.input, outputs=output)

    if balance == True:
        new_model.compile(optimizer='adam', loss=custom_loss_imbalance_visus, metrics=['MeanAbsoluteError'])
    else:
        new_model.compile(optimizer='adam', loss=custom_loss_visus, metrics=['MeanAbsoluteError'])

    new_model.fit([images_train, X_train], Y_train, epochs=n_epochs, batch_size=1)

    new_model.save(f'{modelname}_visus.h5')

if __name__ == "__main__":
    if len(sys.argv) != 6:
        sys.exit("Usage: python trainer.py n_epochs balance aug noruler modelname")
    
    n_epochs = int(sys.argv[1])
    if sys.argv[2] == 'True':
        balance = True
    else:
        balance = False
    
    if sys.argv[3] == 'True':
        aug = True
    else:
        aug = False

    if sys.argv[4] == 'True':
        noruler = True
    else:
        noruler = False

    modelname = sys.argv[5]

    print(f'\nn_epochs: {n_epochs}')
    print(f'balance: {balance}')
    print(f'aug: {aug}')
    print(f'noruler: {noruler}')
    print(f'modelname: {modelname}\n')

    train_general(n_epochs, balance, aug, noruler, modelname)
    eval_general(modelname, balance, noruler, False)
    visualize_attention_general(modelname, noruler, balance, False)
    train_visus(modelname, balance)
    eval_general(f'{modelname}_visus', balance, noruler, True)
    visualize_attention_general(f'{modelname}_visus', noruler, balance, True)

# Addestramento senza bilanciamento
# python trainer.py 10 False True False hybrid_model
    
# Addestramento con bilanciamento
# python trainer.py 10 True True False hybrid_model_bal
    
# Addestramento senza bilanciamento e senza ruler
# python trainer.py 10 False True True hybrid_model_bal_noruler
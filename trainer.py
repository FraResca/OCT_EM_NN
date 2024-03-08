from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy, MeanSquaredError
from tensorflow.keras.models import Model, load_model
from sklearn.utils import shuffle
from scipy.ndimage import zoom
from models import hybnet_multi
from predata import data_prep_multi, splits, data_aug
import matplotlib.pyplot as plt
import numpy as np
import sys

def custom_loss(y_true, y_pred):
    binary_loss = BinaryCrossentropy()(y_true[:,:2], y_pred[:,:2])
    mse_loss = MeanSquaredError()(y_true[:,2:], y_pred[:,2:])
    return binary_loss + mse_loss

def train_hybrid_multi(n_epochs, aug):
    hybrid_model = hybnet_multi()
    hybrid_model.summary()

    X, images, Y = data_prep_multi()
    X, images, Y = shuffle(X, images, Y, random_state=42)
    if aug:
        X, images, Y = data_aug(X, images, Y)

    X_train, X_test, images_train, images_test, Y_train, Y_test = splits(X, images, Y)

    lr = 0.001
    hybrid_model.compile(optimizer=Adam(learning_rate=lr), loss=custom_loss, metrics=['mean_absolute_error'])
    hybrid_model.fit([images_train, X_train], Y_train, epochs=n_epochs, batch_size=2)

    hybrid_model.save('hybrid_model.h5')

    hybrid_model.evaluate([images_test, X_test], Y_test)

def visualize_attention():
    X, images, Y = data_prep_multi()
    _, X_test, _, images_test, _, _ = splits(X, images, Y)

    hybrid_model = load_model('hybrid_model.h5', custom_objects={'custom_loss': custom_loss})

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
    num_cols = 2
    num_rows = num_images // num_cols

    if num_images % num_cols:
        num_rows += 1

    fig = plt.figure(figsize=(10, num_rows * 5))

    for i, (img, att_map) in enumerate(zip(images_test, attention_map_resized), start=1):
        ax = fig.add_subplot(num_rows, num_cols, i)
        ax.imshow(img, cmap='gray')
        ax.imshow(att_map, cmap='jet', alpha=0.5)

    plt.savefig('attention_maps.png')
    plt.show()

if __name__ == '__main__':
    if len(sys.argv) != 4:
        print('Usage: python trainer.py <number of epochs> <augmented yes/no> <vis yes/no>')
        sys.exit(1)
    if sys.argv[2] == 'yes':
        aug = True
    else:
        aug = False
    train_hybrid_multi(int(sys.argv[1]), aug)
    if sys.argv[3] == 'yes':
        visualize_attention()
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model, load_model
from scipy.ndimage import zoom
from models import hybnet
from predata import data_prep
import matplotlib.pyplot as plt
import numpy as np

def train_hybrid():
    hybrid_model = hybnet()
    hybrid_model.summary()

    X_train, X_test, images_train, images_test, y_train, y_test = data_prep()

    lr = 0.001
    hybrid_model.compile(optimizer=Adam(learning_rate=lr), loss='mean_squared_error', metrics=['mean_absolute_error'])
    hybrid_model.fit([images_train, X_train], y_train, epochs=20, batch_size=1)

    hybrid_model.save('hybrid_model.h5')

    hybrid_model.evaluate([images_test, X_test], y_test)

def visualize_attention():
    _, X_test, _, images_test, _, _ = data_prep()

    hybrid_model = load_model('hybrid_model.h5')

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
    train_hybrid()
    visualize_attention()
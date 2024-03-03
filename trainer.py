from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model, load_model
from models import hybnet, convnet, densenet
from predata import data_prep
import matplotlib.pyplot as plt
import numpy as np

def train_hybrid():
    hybrid_model = hybnet()
    hybrid_model.summary()

    [images, X_norms], y_train = data_prep()

    lr = 0.001
    hybrid_model.compile(optimizer=Adam(learning_rate=lr), loss='mean_squared_error', metrics=['mean_absolute_error'])
    hybrid_model.fit([images, X_norms], y_train, epochs=20, batch_size=1)

    hybrid_model.save('hybrid_model.h5')

def visualize_attention():
    hybrid_model = load_model('hybrid_model.h5')
    [images, X_norms], _ = data_prep()

    intermediate_layer_model = Model(inputs=hybrid_model.input,
                                    outputs=hybrid_model.get_layer('multiply').output)

    intermediate_output = intermediate_layer_model.predict([images, X_norms])

    print(intermediate_output.shape)

train_hybrid()
#visualize_attention()
from tensorflow.keras.optimizers import Adam
from models import hybnet, convnet, densenet
from predata import data_prep

hybrid_model = hybnet()
hybrid_model.summary()

[images, X_norm], y_train = data_prep()

lr = 0.001
hybrid_model.compile(optimizer=Adam(learning_rate=lr), loss='mean_squared_error', metrics=['mean_absolute_error'])
hybrid_model.fit([images, X_norm], y_train, epochs=10, batch_size=10)

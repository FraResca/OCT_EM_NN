from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Multiply, BatchNormalization, Conv2D, LocallyConnected2D, GlobalAveragePooling2D, Lambda, Dense, Concatenate, Flatten
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input
import numpy as np
import matplotlib.pyplot as plt
import visualkeras
import cv2

# Definizione strati rete
prevgg16 = VGG16(weights='imagenet', include_top=False, input_shape=(512, 512, 3))
batchnorm = BatchNormalization()
conv1 = Conv2D(256, (3, 3), activation='relu', padding='valid', strides=(2, 2))
conv2 = Conv2D(64, (3, 3), activation='relu', padding='valid', strides=(2, 2))
#conv3 = Conv2D(16, (3, 3), activation='relu', padding='same')
loccon = LocallyConnected2D(1, (3, 3), activation='relu', padding='valid')
conv4 = Conv2D(512, (3, 3), activation='relu', padding='same')
globavpool = GlobalAveragePooling2D()

# Costruzione modello sequenziale fino all'attention mechanism che non è possibile implementare con Squential
'''
model = Sequential()
model.add(prevgg16)
model.add(batchnorm)
model.add(conv1)
model.add(conv2)
model.add(loccon)
model.add(conv4)
'''

# Costruzione modello convoluzionale funzionale
input_img = Input(shape=(512, 512, 3))
conv = prevgg16(input_img)
prevgg16_output = conv
conv = batchnorm(conv)
conv = conv1(conv)
conv = conv2(conv)
#conv = conv3(conv)
conv = loccon(conv)
conv = conv4(conv)
postcnn = conv
conv = Multiply()([conv, prevgg16_output])
conv = globavpool(conv)
y = globavpool(postcnn)
conv = Lambda(lambda tensors: tensors[0] / tensors[1])([conv, y])
convput = conv

conv_model = Model(input_img, convput)

conv_model.summary()

# Costruzione modello fully connected funzionale
input_dense = Input(shape=(12,))
dense = Dense(14, activation='relu')(input_dense)
dense = Dense(6, activation='sigmoid')(dense)
denseput = dense

dense_model = Model(input_dense, denseput)

dense_model.summary()

# Concatenazione modelli
concat = Concatenate()([convput, denseput])
dense_fin = Dense(1024, activation='relu')(concat)
output = Dense(1, activation='sigmoid')(dense_fin)

model = Model(inputs=[input_img, input_dense], outputs=output)

model.summary()

visualkeras.layered_view(model).save('model.png')  # write to disk
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

#print(features.shape)

#visualkeras.layered_view(model).save('model.png')  # write to disk

attention_model = Model(inputs=model.inputs, outputs=model.get_layer('multiply').output)

# Assume `img` is your test image
attention_map = attention_model.predict(img_array)

print(attention_map[0, :, :, 0].shape)

plt.imshow(attention_map[0, :, :, 0], cmap='hot')
plt.colorbar()
plt.savefig('attention_map.png')

# Resize the attention map to the size of the original image
resized_attention_map = cv2.resize(attention_map[0, :, :, 0], (img_array.shape[1], img_array.shape[0]))

# Superimpose the attention map on the original image
superimposed_img = img_array * resized_attention_map[..., np.newaxis]

# Display the original image and the superimposed image
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title('Original Image')
plt.imshow(img_array[0].astype('uint8'))
plt.subplot(1, 2, 2)
plt.title('Image with Attention Map')
plt.imshow((superimposed_img[0] * 255).astype('uint8'))
plt.show()

'''
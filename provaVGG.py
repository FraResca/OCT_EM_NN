from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
from tensorflow.keras.applications.vgg16 import preprocess_input
import numpy as np
import matplotlib.pyplot as plt

#prevgg16 = VGG16(weights='imagenet', include_top=False)
#model = Sequential()
#model.add(prevgg16)

model = VGG16(weights='imagenet', include_top=False)

layer_name = 'block5_conv1'
layer = model.get_layer(layer_name)

feature_extractor_model = Model(inputs=model.input, outputs=layer.output)

img = image.load_img('02_0_cropped.tif', target_size=(512,512))

img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array = preprocess_input(img_array)

feature_map = feature_extractor_model.predict(img_array)

feature_maps = []
for i in range(16):
    feature_maps.append(feature_map[0, :, :, i])

num_rows = 4
num_cols = 4

fig, axes = plt.subplots(num_rows, num_cols, figsize=(12, 8))

axes = axes.flatten()

for i in range(16):
    axes[i].imshow(feature_maps[i], cmap='viridis')  # Assuming feature map is 3D (batch_size, height, width)
    
plt.tight_layout()  # Adjust layout to prevent overlapping
plt.show()
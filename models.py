from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Multiply, BatchNormalization, Conv2D, LocallyConnected2D, GlobalAveragePooling2D, Lambda, Dense, Concatenate, Conv2DTranspose, ZeroPadding2D

def convnet():
    # Definizione strati rete convoluzionale
    prevgg16 = VGG16(weights='vgg16.h5', include_top=False, input_shape=(512, 512, 3))
    for layer in prevgg16.layers:
        layer.trainable = False
    batchnorm = BatchNormalization()
    conv1 = Conv2D(256, (3, 3), activation='relu', padding='valid', strides=(2, 2))
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='valid', strides=(2, 2))
    '''
    conv3 = Conv2D(16, (3, 3), activation='relu', padding='valid', strides=(2, 2))
    '''
    loccon = LocallyConnected2D(1, (3, 3), activation='relu', padding='valid')
    conv4 = Conv2D(512, (3, 3), activation='relu', padding='same')
    globavpool = GlobalAveragePooling2D()

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
    conv = Lambda(lambda tensors: tensors[0] / (tensors[1] + 1e-7))([conv, y])
    convput = conv

    conv_model = Model(input_img, convput)
    return conv_model

def convnet_noruler():
    # Definizione strati rete convoluzionale
    prevgg16 = VGG16(weights='vgg16_noruler.h5', include_top=False, input_shape=(444, 444, 3))
    for layer in prevgg16.layers:
        layer.trainable = False
    batchnorm = BatchNormalization()
    conv1 = Conv2D(256, (3, 3), activation='relu', padding='same')
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')
    conv3 = Conv2D(16, (3, 3), activation='relu', padding='same')
    loccon = LocallyConnected2D(1, (3, 3), activation='relu', padding='valid')
    conv4 = Conv2D(512, (3, 3), activation='relu', padding='same')
    globavpool = GlobalAveragePooling2D()

    # Costruzione modello convoluzionale funzionale
    input_img = Input(shape=(444, 444, 3))
    conv = prevgg16(input_img)
    prevgg16_output = conv
    conv = batchnorm(conv)
    conv = conv1(conv)
    conv = conv2(conv)
    conv = conv3(conv)
    conv = loccon(conv)
    conv = conv4(conv)
    postcnn = conv
    conv = ZeroPadding2D(padding=(1, 1))(conv)
    conv = Multiply()([conv, prevgg16_output])
    conv = globavpool(conv)
    y = globavpool(postcnn)
    conv = Lambda(lambda tensors: tensors[0] / (tensors[1] + 1e-7))([conv, y])
    convput = conv
    conv_model = Model(input_img, convput)
    return conv_model
    
def densenet():
    input_dense = Input(shape=(14,))
    dense = Dense(16, activation='relu')(input_dense)
    dense = Dense(4, activation='sigmoid')(dense)
    denseput = dense

    dense_model = Model(input_dense, denseput)
    return dense_model

def hybnet_multi():
    conv_model = convnet()
    dense_model = densenet()
    concat = Concatenate()([conv_model.output, dense_model.output])
    dense_fin = Dense(1024, activation='relu')(concat)
    output = Dense(4, name='output')(dense_fin)

    hybrid_model = Model(inputs=[conv_model.input, dense_model.input], outputs=output)
    return hybrid_model

def hybnet_multi_noruler():
    conv_model = convnet_noruler()
    dense_model = densenet()
    concat = Concatenate()([conv_model.output, dense_model.output])
    dense_fin = Dense(1024, activation='relu')(concat)
    output = Dense(4, name='output')(dense_fin)

    hybrid_model = Model(inputs=[conv_model.input, dense_model.input], outputs=output)
    return hybrid_model
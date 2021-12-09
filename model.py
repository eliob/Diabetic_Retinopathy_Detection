# from tensorflow.keras.layers import Input  # Input Layer
import numpy as np
from tensorflow.keras import layers
from tensorflow.keras.applications import DenseNet121  # Keras Application
# from keras.applications.inception_v3 import preprocess_input
from keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.layers import Dense  # Dense Layer (Fully connected)
from tensorflow.keras.models import Model  # Model Structure


def get_model(image_size=299, model_type='InceptionV3'):
    input_shape = (image_size, image_size, 3)
    img_input = layers.Input(shape=input_shape)
    if model_type == 'DenseNet121':
        base_model = DenseNet121(include_top=False,
                                 input_tensor=img_input,
                                 input_shape=input_shape,
                                 pooling="max",
                                 weights='imagenet')
        base_model.trainable = True
        x = base_model.output
        predictions = Dense(5, activation="softmax", name="predictions")(x)
        model = Model(inputs=img_input, outputs=predictions)
        return model
    elif model_type == 'InceptionV3':
        base_model = InceptionV3(include_top=False,
                                 input_tensor=img_input,
                                 input_shape=input_shape,
                                 # pooling="max",
                                 weights='imagenet')
        base_model.trainable = True
        x = base_model.output
        x = layers.GlobalAveragePooling2D()(x)
        predictions = Dense(5, activation="softmax", name="predictions")(x)
        model = Model(inputs=img_input, outputs=predictions)
        return model

    elif model_type == 'InceptionV3_att':
        base_model = InceptionV3(include_top=False,
                                 input_tensor=img_input,
                                 input_shape=input_shape,
                                 # pooling="max",
                                 weights='imagenet')
        base_model.trainable = False
        pt_features = base_model(img_input)
        pt_depth = base_model.get_output_shape_at(0)[-1]
        bn_features = layers.BatchNormalization()(pt_features)
        #
        # # here we do an attention mechanism to turn pixels in the GAP on an off
        attn_layer = layers.Conv2D(64, kernel_size=(1, 1), padding='same', activation='relu')(
            layers.Dropout(0.5)(bn_features))
        attn_layer = layers.Conv2D(16, kernel_size=(1, 1), padding='same', activation='relu')(attn_layer)
        attn_layer = layers.Conv2D(8, kernel_size=(1, 1), padding='same', activation='relu')(attn_layer)
        attn_layer = layers.Conv2D(1,
                                   kernel_size=(1, 1),
                                   padding='valid',
                                   activation='sigmoid')(attn_layer)

        # fan it out to all of the channels
        up_c2_w = np.ones((1, 1, 1, pt_depth))
        up_c2 = layers.Conv2D(pt_depth, kernel_size=(1, 1), padding='same',
                              activation='linear', use_bias=False, weights=[up_c2_w])
        up_c2.trainable = False
        attn_layer = up_c2(attn_layer)

        mask_features = layers.multiply([attn_layer, bn_features])
        gap_features = layers.GlobalAveragePooling2D()(mask_features)
        gap_mask = layers.GlobalAveragePooling2D()(attn_layer)

        # to account for missing values from the attention model
        gap = layers.Lambda(lambda x: x[0] / x[1], name='RescaleGAP')([gap_features, gap_mask])
        gap_dr = layers.Dropout(0.25)(gap)
        dr_steps = layers.Dropout(0.25)(Dense(128, activation='relu')(gap_dr))
        out_layer = Dense(5, activation='softmax', name="predictions")(dr_steps)
        retina_model = Model(inputs=[img_input], outputs=[out_layer]) #out_layer
        return retina_model

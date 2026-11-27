import cv2
import numpy as np
from tensorflow import keras

def preprocess_b1(m, target_size=(64, 64)):
    """Exact Phase 2b enc_resize + f32 preprocessing (cv2.INTER_AREA + uint8 quantization)."""
    m = np.asarray(m)
    e = np.stack([m == 0, m == 1, m == 2], axis=-1).astype(np.float32)
    u8 = np.rint(cv2.resize(e, target_size, interpolation=cv2.INTER_AREA) * 255).astype(np.uint8)
    return u8.astype(np.float32) / 255.0

def build_b1_model(num_classes=9, input_shape=(64, 64, 3), dropout_rate=0.3):
    """Exact Phase 2b B1 Resize CNN architecture."""
    L = keras.layers
    x = inp = keras.Input(shape=input_shape)
    for f in (32, 64, 128):
        for _ in range(2):
            x = L.ReLU()(L.BatchNormalization(momentum=0.9)(
                L.Conv2D(f, 3, padding='same', use_bias=False)(x)
            ))
        x = L.MaxPool2D()(x)
    x = L.Dropout(dropout_rate)(L.Dense(256, activation='relu')(L.Flatten()(x)))
    return keras.Model(inp, L.Dense(num_classes, activation='softmax')(x), name='b1_resize_cnn')

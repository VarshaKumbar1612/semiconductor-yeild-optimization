import numpy as np, pandas as pd, tensorflow as tf
from tensorflow import keras
K = 9

class SPP(keras.layers.Layer):
    def __init__(self, levels=(1, 2, 4), **kw):
        super().__init__(**kw); self.levels = tuple(levels)
    def call(self, x):
        H, W_ = tf.shape(x)[1], tf.shape(x)[2]; outs = []
        for n in self.levels:
            for i in range(n):
                h0 = (i * H) // n; h1 = tf.maximum(((i + 1) * H + n - 1) // n, h0 + 1)
                for j in range(n):
                    w0 = (j * W_) // n; w1 = tf.maximum(((j + 1) * W_ + n - 1) // n, w0 + 1)
                    outs.append(tf.reduce_max(x[:, h0:h1, w0:w1, :], axis=[1, 2]))
        return tf.concat(outs, -1)
    def compute_output_shape(self, s): return (s[0], sum(n * n for n in self.levels) * s[-1])
    def get_config(self): return {**super().get_config(), 'levels': list(self.levels)}

def build_spp(drop=0.3):          # final architecture: GN conv blocks, SPP, LayerNorm, Dense(256), head std 0.02
    L = keras.layers; x = inp = keras.Input((None, None, 3))
    for bi, f in enumerate((32, 64, 128)):
        for _ in range(2):
            x = L.ReLU()(L.GroupNormalization(groups=8)(L.Conv2D(f, 3, padding='same', use_bias=False)(x)))
        if bi < 2: x = L.MaxPool2D()(x)
    x = L.LayerNormalization()(SPP()(x))
    x = L.Dropout(drop)(L.Dense(256, activation='relu')(x))
    return keras.Model(inp, L.Dense(K, activation='softmax',
                       kernel_initializer=keras.initializers.RandomNormal(stddev=0.02))(x))

def make_pred(model):
    @tf.function(input_signature=[tf.TensorSpec([None, None, None, 3], tf.float32)])
    def pred(x): return model(x, training=False)
    return pred

def onehot(ms):
    M = np.stack(ms); return np.stack([M == 0, M == 1, M == 2], -1).astype(np.float32)

def predict_native(pred, idx, maps, SK, Hs, Ws, pix=128 * 48 * 48, bs_ref=128):
    P = np.zeros((len(idx), K), np.float32); keys = SK[idx]
    for k in pd.unique(keys):
        sel = np.where(keys == k)[0]; i0 = idx[sel[0]]
        bs = 2 * int(min(bs_ref, max(1, pix // (Hs[i0] * Ws[i0]))))
        for c in range(0, len(sel), bs):
            s = sel[c:c + bs]; P[s] = pred(tf.constant(onehot([maps[i] for i in idx[s]]))).numpy()
    return P

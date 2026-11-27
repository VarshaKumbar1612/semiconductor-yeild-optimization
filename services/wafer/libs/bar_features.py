import numpy as np
from skimage.measure import label as cc_label, regionprops
from skimage.transform import radon


def _rs(v, n):
    return np.interp(np.linspace(0, 1, n), np.linspace(0, 1, len(v)), v)


def mask_feats(m):  # outline only, no fail information: 35 values
    on = m > 0
    H, W_ = on.shape
    return np.concatenate([[H, W_, on.sum()], _rs(on.sum(1) / W_, 16), _rs(on.sum(0) / H, 16)])


def pattern_feats(m):  # 102 values
    on, fail = m > 0, m == 2
    H, W_ = m.shape
    n_on = max(int(on.sum()), 1)
    n_fail = int(fail.sum())
    yy, xx = np.indices(m.shape)
    cy, cx = (H - 1) / 2, (W_ - 1) / 2
    r = np.hypot((yy - cy) / (H / 2), (xx - cx) / (W_ / 2))
    th = (np.arctan2(yy - cy, xx - cx) + np.pi) / (2 * np.pi)
    cell = np.digitize(r, [0.25, 0.5, 0.75, 0.9]) * 8 + np.minimum((th * 8).astype(int), 7)
    on_c = np.bincount(cell[on], minlength=40).astype(float)
    fl_c = np.bincount(cell[fail], minlength=40).astype(float)
    dens = np.divide(fl_c, on_c, out=np.zeros(40), where=on_c > 0)
    ring = fl_c.reshape(5, 8).sum(1) / np.maximum(on_c.reshape(5, 8).sum(1), 1)
    sect = fl_c.reshape(5, 8).sum(0) / np.maximum(on_c.reshape(5, 8).sum(0), 1)
    sino = radon(fail.astype(float), theta=np.linspace(0, 180, 30, endpoint=False), circle=False) / max(H, W_)
    rad = np.concatenate([_rs(sino.mean(1), 20), _rs(sino.std(1), 20)])
    lab = cc_label(fail, connectivity=2)
    if lab.max() > 0:
        p = max(regionprops(lab), key=lambda q: q.area)
        pr, pc_ = p.centroid
        reg = [
            p.area / n_on,
            p.area / max(n_fail, 1),
            p.eccentricity,
            p.solidity,
            p.major_axis_length / max(H, W_),
            p.minor_axis_length / max(H, W_),
            np.hypot((pr - cy) / (H / 2), (pc_ - cx) / (W_ / 2)),
            lab.max(),
        ]
    else:
        reg = [0.0] * 8
    return np.concatenate([[n_fail / n_on], ring, sect, dens, rad, reg])


def all_feats(m):  # 35 + 102 = 137 values, float32
    m = np.asarray(m)
    return np.concatenate([mask_feats(m), pattern_feats(m)]).astype(np.float32)
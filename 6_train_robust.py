"""Train the 7-class (6 defects + background) CNN with a BOARD-LEVEL split.

Validation boards come from whole held-out DeepPCB groups, so no template is shared with training.
Output: pcb_cnn_model_v2.keras (original pcb_cnn_model.keras is untouched).
Run from inside this folder:  python 6_train_robust.py
"""
import glob, os, random
import cv2, numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from pcb_common import candidates, CLASS_NAMES, BG, LABEL_TO_IDX, prep

VAL_GROUPS = ["group50600", "group90100"]   # held out entirely
random.seed(0); np.random.seed(0); tf.random.set_seed(0)
R = lambda im: cv2.resize(im, (128, 128), interpolation=cv2.INTER_AREA)


def read_boxes(p):
    out = []
    for line in open(p):
        v = line.split()
        if len(v) == 5:
            out.append(tuple(map(int, v)))
    return out


def synth_bg(n):
    """Non-board junk: flat, gradient, noisy, blobby, stripe patches (table, skin, shadows, text-ish)."""
    out = []
    for _ in range(n):
        base = np.random.randint(0, 256)
        im = np.full((128, 128), base, np.float32)
        k = random.choice(["flat", "grad", "noise", "blob", "stripe"])
        if k == "grad":
            im += np.linspace(0, np.random.randint(-120, 120), 128)[None, :] * np.random.choice([0, 1])
            im += np.linspace(0, np.random.randint(-120, 120), 128)[:, None]
        elif k == "noise":
            im += np.random.randn(128, 128) * np.random.randint(5, 40)
        elif k == "blob":
            for _ in range(random.randint(1, 4)):
                cv2.circle(im, tuple(np.random.randint(0, 128, 2).tolist()), random.randint(8, 60),
                           float(np.random.randint(0, 256)), -1)
            im = cv2.GaussianBlur(im, (0, 0), random.uniform(0, 6))
        elif k == "stripe":
            for _ in range(random.randint(1, 6)):
                cv2.line(im, tuple(np.random.randint(0, 128, 2).tolist()), tuple(np.random.randint(0, 128, 2).tolist()),
                         float(np.random.randint(0, 256)), random.randint(1, 12))
        out.append(np.clip(im, 0, 255).astype(np.uint8))
    return out


def win(img, cx, cy, side):
    h, w = img.shape; s = int(max(24, side)); x = int(np.clip(cx - s / 2, 0, max(0, w - s))); y = int(np.clip(cy - s / 2, 0, max(0, h - s)))
    return R(img[y:y + s, x:x + s])


def build(groups):
    """Window-style crops (same geometry as inference sliding windows)."""
    X, y = [], []
    for g in groups:
        for tp in glob.glob(f"PCBData/{g}/*/*_test.jpg"):
            bid = os.path.basename(tp)[:-9]
            lp = glob.glob(f"PCBData/{g}/*/{bid}.txt")
            test = cv2.imread(tp, 0); tmp = cv2.imread(tp.replace("_test", "_temp"), 0)
            if test is None or tmp is None or not lp:
                continue
            boxes = [b for b in read_boxes(lp[0]) if b[4] in LABEL_TO_IDX]
            H, W = test.shape
            for x1, y1, x2, y2, t in boxes:
                for _ in range(3):
                    side = max(x2 - x1, y2 - y1) * random.uniform(1.4, 2.4)
                    cx = (x1 + x2) / 2 + random.uniform(-.2, .2) * side; cy = (y1 + y2) / 2 + random.uniform(-.2, .2) * side
                    X.append(win(test, cx, cy, max(side, 40))); y.append(LABEL_TO_IDX[t])
            for _ in range(8):   # defect-free template windows
                side = random.uniform(40, 80); X.append(win(tmp, random.uniform(0, W), random.uniform(0, H), side)); y.append(BG)
            n = 0
            while n < 6:         # test-image windows far from every defect
                side = random.uniform(40, 80); cx, cy = random.uniform(0, W), random.uniform(0, H)
                if any(a - .6 * side < cx < c + .6 * side and b - .6 * side < cy < d + .6 * side for a, b, c, d, _ in boxes):
                    if random.random() < .9: continue
                X.append(win(test, cx, cy, side)); y.append(BG); n += 1
    return X, y


def augment(im):
    im = im.astype(np.float32)
    if random.random() < .5: im = 255 - im                       # polarity (colour boards map differently to grey)
    if random.random() < .5: im = np.fliplr(im)
    if random.random() < .5: im = np.flipud(im)
    im = np.rot90(im, random.randint(0, 3))
    a = random.uniform(.5, 1.5); b = random.uniform(-60, 60)    # contrast / brightness
    im = im * a + b
    im = 255 * (np.clip(im, 0, 255) / 255) ** random.uniform(.6, 1.6)   # gamma
    if random.random() < .5: im = cv2.GaussianBlur(im, (0, 0), random.uniform(.3, 2.2))
    if random.random() < .5: im = im + np.random.randn(*im.shape) * random.uniform(2, 14)
    if random.random() < .4:                                     # resolution loss
        s = random.randint(24, 80); im = cv2.resize(cv2.resize(im, (s, s)), (128, 128))
    M = cv2.getRotationMatrix2D((64, 64), random.uniform(-15, 15), random.uniform(.85, 1.15))
    im = cv2.warpAffine(im, M, (128, 128), borderMode=cv2.BORDER_REFLECT)
    return np.clip(im, 0, 255).astype(np.uint8)


class Seq(tf.keras.utils.Sequence):
    def __init__(self, X, y, train, bs=64):
        super().__init__(); self.X, self.y, self.train, self.bs = X, np.array(y), train, bs
        self.idx = np.arange(len(X)); self.on_epoch_end()
    def __len__(self): return int(np.ceil(len(self.X) / self.bs))
    def on_epoch_end(self):
        if self.train: np.random.shuffle(self.idx)
    def __getitem__(self, i):
        ids = self.idx[i * self.bs:(i + 1) * self.bs]
        xs = [augment(self.X[j]) if self.train else self.X[j] for j in ids]
        return np.stack([prep(x) for x in xs]), self.y[ids]


if __name__ == "__main__":
    all_groups = sorted(os.path.basename(p) for p in glob.glob("PCBData/group*"))
    tr_groups = [g for g in all_groups if g not in VAL_GROUPS]
    Xtr, ytr = build(tr_groups); Xva, yva = build(VAL_GROUPS)
    s = synth_bg(1500); Xtr += s; ytr += [BG] * len(s)
    s = synth_bg(300); Xva += s; yva += [BG] * len(s)
    print("train groups:", tr_groups, "val groups:", VAL_GROUPS)
    print("train counts:", dict(zip(CLASS_NAMES, np.bincount(ytr, minlength=7))))
    print("val counts:  ", dict(zip(CLASS_NAMES, np.bincount(yva, minlength=7))))
    # background is the most diverse class -> weight it up a little
    cw = {i: 1.0 for i in range(7)}; cw[BG] = 1.5
    model = models.Sequential([
        layers.Rescaling(1. / 255, input_shape=(128, 128, 1)),
        layers.Conv2D(32, 3, activation="relu"), layers.MaxPooling2D(),
        layers.Conv2D(64, 3, activation="relu"), layers.MaxPooling2D(),
        layers.Conv2D(64, 3, activation="relu"), layers.MaxPooling2D(),
        layers.Flatten(), layers.Dense(64, activation="relu"), layers.Dropout(0.5),
        layers.Dense(7, activation="softmax")])
    model.compile("adam", "sparse_categorical_crossentropy", metrics=["accuracy"])
    model.fit(Seq(Xtr, ytr, True), validation_data=Seq(Xva, yva, False), epochs=12, class_weight=cw,
              callbacks=[tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True)])
    model.save("pcb_cnn_model_v2.keras")
    print("saved pcb_cnn_model_v2.keras")

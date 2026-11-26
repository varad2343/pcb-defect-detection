"""Before/after on HELD-OUT boards (whole groups unseen by pcb_cnn_model_v2).

Sets:  gray  = original DeepPCB test images
       photo = SYNTHETIC colour proxy: board recoloured, lit, blurred, pasted on a cluttered table.
               This is a stand-in, NOT real data. Pass --real DIR for your own colour PCB photos
               (reports detections per image + how many fall outside the localized board).
Baseline = old model, old app pipeline (no localization, raw gray, 6 classes, conf 0.95).
Run: python 7_evaluate.py [--n 60] [--real DIR]
"""
import argparse, glob, os, random
import cv2, numpy as np
from tensorflow import keras
from pcb_common import candidates, detect, localize_board, CLASS_NAMES, LABEL_TO_IDX

OLD_NAMES = ["copper", "mousebite", "open", "pin-hole", "short", "spur"]
random.seed(1); np.random.seed(1)


def detect_old(model, bgr, conf=0.95):
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    out = []
    for x, y, w, h in candidates(gray):
        r = cv2.resize(gray[y:y + h, x:x + w], (128, 128), interpolation=cv2.INTER_AREA).astype(np.float32)
        p = model.predict(r[None, ..., None], verbose=0)[0]
        if p.max() >= conf:
            out.append((x, y, w, h, OLD_NAMES[int(p.argmax())], float(p.max())))
    return out


def photo_proxy(gray, boxes):
    """Recolour + light + clutter. Returns bgr, new boxes, board rect (x,y,w,h)."""
    H, W = 1100, 1400
    bg = np.full((H, W, 3), [random.randint(40, 220) for _ in range(3)], np.float32)
    bg += np.random.randn(H, W, 1) * random.uniform(4, 20)
    bg = cv2.GaussianBlur(bg, (0, 0), 2)
    for _ in range(random.randint(10, 25)):                       # clutter: tools, hands, cables, text
        c = tuple(float(v) for v in np.random.randint(0, 256, 3))
        k = random.random()
        p1 = (random.randint(0, W), random.randint(0, H))
        if k < .35: cv2.circle(bg, p1, random.randint(15, 90), c, -1)
        elif k < .7: cv2.rectangle(bg, p1, (p1[0] + random.randint(20, 200), p1[1] + random.randint(10, 120)), c, random.choice([-1, 3]))
        else: cv2.line(bg, p1, (random.randint(0, W), random.randint(0, H)), c, random.randint(2, 14))
    s = random.uniform(.9, 1.5)
    bw, bh = int(gray.shape[1] * s), int(gray.shape[0] * s)
    g = cv2.resize(gray, (bw, bh)).astype(np.float32) / 255
    hue = random.choice([55, 60, 105, 115, 2, 175, 20])
    sat = random.randint(90, 200) if random.random() < .85 else random.randint(0, 25)
    hsv = np.stack([np.full_like(g, hue), np.full_like(g, sat), 40 + g * random.uniform(150, 215)], -1).astype(np.uint8)
    board = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR).astype(np.float32)
    board *= (np.linspace(random.uniform(.6, 1), 1, bw)[None, :, None])  # lighting gradient
    board = cv2.GaussianBlur(board, (0, 0), random.uniform(.3, 1.5))
    ox, oy = random.randint(20, W - bw - 20), random.randint(20, H - bh - 20)
    bg[oy:oy + bh, ox:ox + bw] = board
    bg += np.random.randn(H, W, 3) * random.uniform(1, 8)
    nb = [(int(a * s) + ox, int(b * s) + oy, int(c * s) + ox, int(d * s) + oy) for a, b, c, d in boxes]
    return np.clip(bg, 0, 255).astype(np.uint8), nb, (ox, oy, bw, bh)


def score(dets, gts, rect=None, pad=12):
    """recall (centre of a detection inside a GT box), FP count, FP outside the board rect."""
    hit = 0
    for a, b, c, d in gts:
        if any(a - pad <= x + w / 2 <= c + pad and b - pad <= y + h / 2 <= d + pad for x, y, w, h, *_ in dets):
            hit += 1
    fp = out = 0
    for x, y, w, h, *_ in dets:
        cx, cy = x + w / 2, y + h / 2
        if not any(a - pad <= cx <= c + pad and b - pad <= cy <= d + pad for a, b, c, d in gts):
            fp += 1
            if rect and not (rect[0] <= cx <= rect[0] + rect[2] and rect[1] <= cy <= rect[1] + rect[3]):
                out += 1
    return hit, len(gts), fp, out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=60); ap.add_argument("--real")
    a = ap.parse_args()
    old = keras.models.load_model("pcb_cnn_model.keras"); new = keras.models.load_model("pcb_cnn_model_v2.keras")
    tests = [t for g in ("group50600", "group90100") for t in glob.glob(f"PCBData/{g}/*/*_test.jpg")]
    random.shuffle(tests); tests = tests[:a.n]
    tot = {k: np.zeros(4) for k in ("old_gray", "new_gray", "old_photo", "new_photo")}
    for tp in tests:
        gray = cv2.imread(tp, 0)
        lp = tp.replace("_test.jpg", ".txt").replace("PCBData/", "PCBData/").replace("\\", "/")
        lp = glob.glob(os.path.join(os.path.dirname(tp), os.path.basename(tp)[:-9] + ".txt"))
        if not lp:  # labels live in the sibling "<id>_not" folder
            lp = glob.glob(os.path.dirname(os.path.dirname(tp)) + "/*_not/" + os.path.basename(tp)[:-9] + ".txt")
        gts = [tuple(map(int, l.split()[:4])) for l in open(lp[0]) if len(l.split()) == 5]
        bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        tot["old_gray"] += score(detect_old(old, bgr), gts)
        tot["new_gray"] += score(detect(new, bgr, use_localization=False), gts)
        pb, pg, rect = photo_proxy(gray, gts)
        tot["old_photo"] += score(detect_old(old, pb), pg, rect)
        tot["new_photo"] += score(detect(new, pb), pg, rect)
    print(f"\nBoards evaluated: {len(tests)} (held-out groups)\n")
    print(f"{'pipeline / set':<18}{'recall':>8}{'FP/img':>9}{'FP outside board/img':>24}")
    for k, v in tot.items():
        print(f"{k:<18}{v[0] / max(v[1], 1):>8.1%}{v[2] / len(tests):>9.2f}{v[3] / len(tests):>24.2f}")
    if a.real:
        print("\nReal photos (no labels): detections inside / outside localized board")
        for p in sorted(glob.glob(os.path.join(a.real, "*"))):
            im = cv2.imread(p)
            if im is None: continue
            d_old, d_new = detect_old(old, im), detect(new, im)
            print(f"{os.path.basename(p):<30} old={len(d_old):>3}  new={len(d_new):>3}  board_found={localize_board(im) is not None}")


if __name__ == "__main__":
    main()

"""Shared preprocessing for training AND inference (one code path, no drift).

Pipeline for a photo: localize board -> crop + rescale to DeepPCB scale (640 px wide)
-> contour candidates -> prep() each candidate -> CNN (7 classes, last = background).
"""
import cv2
import numpy as np

CLASS_NAMES = ["copper", "mousebite", "open", "pin-hole", "short", "spur", "background"]
BG = 6
LABEL_TO_IDX = {1: 2, 2: 4, 3: 1, 4: 5, 5: 0, 6: 3}  # DeepPCB label id -> CLASS_NAMES index
BOARD_W = 640  # DeepPCB image width; candidate size limits below assume this scale


def prep(gray_crop):
    """128x128 float32, contrast-stretched per crop so lighting/colour polarity gaps shrink."""
    g = cv2.resize(gray_crop, (128, 128), interpolation=cv2.INTER_AREA).astype(np.float32)
    lo, hi = np.percentile(g, (2, 98))
    g = np.clip((g - lo) / max(hi - lo, 8.0), 0, 1) * 255.0
    return g[..., None]


def candidates(gray, mask=None):
    """Same contour candidate logic as app.py, optionally restricted to the board mask."""
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    th = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    if mask is not None:
        th = cv2.bitwise_and(th, mask)
    cs, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = []
    for c in cs:
        x, y, w, h = cv2.boundingRect(c)
        if not (20 < w < 300 and 20 < h < 300):
            continue
        per = cv2.arcLength(c, True)
        circ = 4 * np.pi * cv2.contourArea(c) / (per * per) if per > 0 else 0.0
        if circ > 0.75:
            continue
        out.append((x, y, w, h))
    return out


def windows(gray, mask=None, sizes=(48, 72), stride=24):
    """Sliding square windows over the board (replaces contour candidates, which missed ~60% of defects)."""
    H, W = gray.shape
    out = []
    for s in sizes:
        for y in range(0, H - s + 1, stride):
            for x in range(0, W - s + 1, stride):
                if mask is not None and mask[y + s // 2, x + s // 2] == 0:
                    continue
                out.append((x, y, s, s))
    return out


def nms(dets, thr=0.3):
    """dets: (x,y,w,h,name,conf). Greedy by confidence."""
    keep = []
    for d in sorted(dets, key=lambda t: -t[5]):
        ok = True
        for k in keep:
            ix = max(0, min(d[0] + d[2], k[0] + k[2]) - max(d[0], k[0]))
            iy = max(0, min(d[1] + d[3], k[1] + k[3]) - max(d[1], k[1]))
            if ix * iy / (d[2] * d[3] + k[2] * k[3] - ix * iy) > thr:
                ok = False; break
        if ok:
            keep.append(d)
    return keep


def localize_board(bgr):
    """Return (x, y, w, h, mask) of the board in the photo, or None if no board found.

    Colour boards: the solder-mask hue is the saturation-weighted hue peak.
    Dark/grey boards: Otsu on grey. Largest blob wins, holes filled.
    ponytail: classical heuristic; swap for a trained board segmenter if it fails on your photos.
    """
    h, w = bgr.shape[:2]
    s = 640.0 / max(h, w)
    small = cv2.resize(bgr, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(cv2.GaussianBlur(small, (5, 5), 0), cv2.COLOR_BGR2HSV)
    hue, sat, val = cv2.split(hsv)
    sat_ok = (sat > 60) & (val > 40)
    cand = []
    if sat_ok.mean() > 0.03:
        hist = np.bincount(hue[sat_ok].ravel(), minlength=180).astype(np.float32)
        hist = np.convolve(np.r_[hist[-8:], hist, hist[:8]], np.ones(17), "same")[8:-8]
        peak = int(np.argmax(hist))
        d = np.minimum(np.abs(hue.astype(int) - peak), 180 - np.abs(hue.astype(int) - peak))
        cand.append(((d < 14) & sat_ok).astype(np.uint8) * 255)
    g = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    _, otsu = cv2.threshold(cv2.GaussianBlur(g, (7, 7), 0), 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    cand.append(otsu)
    best = None
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    for m in cand:
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k, iterations=2)
        cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cs:
            continue
        c = max(cs, key=cv2.contourArea)
        frac = cv2.contourArea(c) / m.size
        if 0.08 < frac < 0.97 and (best is None or frac > best[0]):
            best = (frac, c)
    if best is None:
        return None
    mask = np.zeros(g.shape, np.uint8)
    cv2.drawContours(mask, [cv2.convexHull(best[1])], -1, 255, -1)
    mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
    x, y, bw, bh = cv2.boundingRect(mask)
    return x, y, bw, bh, mask


def detect(model, bgr, conf=0.50, use_localization=True):
    """Full pipeline. Returns list of (x, y, w, h, class_name, confidence) in ORIGINAL image coords."""
    H, W = bgr.shape[:2]
    mask_full = None
    ox = oy = 0
    img = bgr
    if use_localization:
        loc = localize_board(bgr)
        if loc is None:
            return []  # no board in frame -> nothing to report
        x, y, bw, bh, mask_full = loc
        img, ox, oy = bgr[y:y + bh, x:x + bw], x, y
        mask_full = mask_full[y:y + bh, x:x + bw]
    sc = BOARD_W / img.shape[1] if use_localization else 1.0
    img_s = cv2.resize(img, None, fx=sc, fy=sc, interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(img_s, cv2.COLOR_BGR2GRAY)
    m = None
    if mask_full is not None:
        m = cv2.erode(cv2.resize(mask_full, (gray.shape[1], gray.shape[0]), interpolation=cv2.INTER_NEAREST),
                      np.ones((9, 9), np.uint8))
    boxes = windows(gray, m)
    if not boxes:
        return []
    batch = np.stack([prep(gray[y:y + h, x:x + w]) for x, y, w, h in boxes])
    probs = model.predict(batch, verbose=0)
    out = []
    for (x, y, w, h), p in zip(boxes, probs):
        i = int(np.argmax(p))
        if i == BG or p[i] < conf:
            continue
        out.append((int(x / sc) + ox, int(y / sc) + oy, int(w / sc), int(h / sc), CLASS_NAMES[i], float(p[i])))
    return nms(out)

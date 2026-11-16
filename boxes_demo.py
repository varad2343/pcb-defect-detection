import sys, cv2, time
from tensorflow import keras
from pcb_common import detect
from gemini_review import find_defects
src, dst = sys.argv[1], sys.argv[2]
im = cv2.imread(src); m = keras.models.load_model("pcb_cnn_model_v2.keras")
o = im.copy()
for x, y, w, h, n, c in detect(m, im, 0.5):
    cv2.rectangle(o, (x, y), (x + w, y + h), (0, 0, 255), 2); cv2.putText(o, f"{n} {c:.0%}", (x, max(y - 5, 12)), 0, 0.5, (0, 0, 255), 2)
t = time.time(); g = find_defects(im); print("scan seconds", round(time.time() - t, 1))
for x, y, w, h, n, c, why in g:
    cv2.rectangle(o, (x, y), (x + w, y + h), (0, 120, 255), 3); cv2.putText(o, f"{n} {c:.0%}", (x, max(y - 6, 14)), 0, 0.6, (0, 120, 255), 2); print(n, round(c, 2), (x, y, w, h), why)
cv2.imwrite(dst, o); print("done")

"""Optional detailed defect write-up from Gemini."""
import os, cv2

PROMPT = """You are a PCB quality inspector. The image shows a printed circuit board with red boxes from an automatic detector
(label + confidence). The detector is unreliable on assembled boards: solder joints, pads and components are often wrongly flagged.
For EACH box: say whether it is a REAL defect (open, short, mousebite, spur, spurious copper, pin-hole, scratch, burn, corrosion,
cracked trace, solder bridge) or a FALSE ALARM, with a one-line reason. Then list any visible defects the detector missed.
End with an overall verdict: PASS / FAIL / UNSURE and the main risk. Be concrete about location (e.g. top-left, near the connector).
Do not invent defects you cannot see."""


def review(bgr_annotated, dets):
    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        return "Detailed review is not available right now."
    from google import genai
    from google.genai import types
    ok, buf = cv2.imencode(".jpg", bgr_annotated, [cv2.IMWRITE_JPEG_QUALITY, 90])
    listing = "\n".join(f"- {n} {c:.0%} at x={x},y={y},w={w},h={h}" for x, y, w, h, n, c in dets) or "- (no boxes)"
    client = genai.Client(api_key=key, http_options=types.HttpOptions(timeout=90000))
    r = client.models.generate_content(
        model=os.environ.get("GEMINI_MODEL", "gemini-3.5-flash"),
        contents=[types.Part.from_bytes(data=buf.tobytes(), mime_type="image/jpeg"), PROMPT + "\nDetector boxes:\n" + listing])
    return r.text


FIND_PROMPT = """You are a PCB defect inspector. Find every visible fault on the printed circuit board in this image.
Use ONLY these 6 classes: copper (spurious copper), mousebite, open (open circuit), pin-hole, short (short circuit), spur. Report nothing else.
Return a JSON list; each item: {"box_2d":[ymin,xmin,ymax,xmax] normalized 0-1000, "label": one class, "confidence": 0-1, "reason": short sentence}.
Report only real faults of these 6 types that you can actually see on the board (not normal pads, solder joints, silkscreen, or the background). If none, return []."""


def find_defects(bgr):
    """Gemini fault scan -> same tuple format as the CNN: (x, y, w, h, name, conf) plus reason, in image pixels."""
    import json
    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        return []
    from google import genai
    from google.genai import types
    H, W = bgr.shape[:2]
    sm = bgr if max(H, W) <= 1280 else cv2.resize(bgr, None, fx=1280 / max(H, W), fy=1280 / max(H, W), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", sm, [cv2.IMWRITE_JPEG_QUALITY, 90])
    client = genai.Client(api_key=key, http_options=types.HttpOptions(timeout=90000))
    r = client.models.generate_content(
        model=os.environ.get("GEMINI_MODEL", "gemini-3.5-flash"),
        contents=[types.Part.from_bytes(data=buf.tobytes(), mime_type="image/jpeg"), FIND_PROMPT],
        config=types.GenerateContentConfig(response_mime_type="application/json", temperature=0.1))
    out = []
    ok6 = {"copper", "mousebite", "open", "pin-hole", "short", "spur"}
    for d in json.loads(r.text):
        if str(d.get("label", "")).lower().replace("_", "-").replace("pinhole", "pin-hole") not in ok6:
            continue
        y1, x1, y2, x2 = d["box_2d"]
        x, y = int(x1 / 1000 * W), int(y1 / 1000 * H)
        out.append((x, y, max(int((x2 - x1) / 1000 * W), 4), max(int((y2 - y1) / 1000 * H), 4),
                    str(d["label"]).lower().replace("_", "-").replace("pinhole", "pin-hole"), float(d.get("confidence", 0.5)), str(d.get("reason", ""))))
    return out

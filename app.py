"""Countersign - signature comparison tool.  Run: python app.py  ->  http://127.0.0.1:5000"""
import base64
import cv2
import numpy as np
from flask import Flask, jsonify, render_template, request
from skimage.metrics import structural_similarity

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 12 * 1024 * 1024
W, H = 300, 150  # normalised size for comparison

WEIGHTS = {"structure": .25, "shape": .20, "strokes": .25, "keypoints": .15, "proportion": .15}


def load_ink(data: bytes):
    """Decode, binarise (ink=255), denoise and crop to the signature."""
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError("Could not read the image. Use PNG, JPG, BMP or WEBP.")
    if max(img.shape) > 1600:
        s = 1600 / max(img.shape)
        img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    img = cv2.GaussianBlur(img, (3, 3), 0)
    _, ink = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    if ink.mean() > 127:  # light ink on dark paper
        ink = 255 - ink
    n, lab, stats, _ = cv2.connectedComponentsWithStats(ink, 8)
    clean = np.zeros_like(ink)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= 12:
            clean[lab == i] = 255
    ys, xs = np.where(clean > 0)
    if len(xs) < 30:
        raise ValueError("No signature found. Use a clear image of dark ink on lighter paper.")
    crop = clean[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    h, w = crop.shape
    return crop, w / h, float((crop > 0).mean())


def normalise(crop):
    return cv2.resize(crop, (W, H), interpolation=cv2.INTER_AREA)


def corr(a, b):
    if a.std() == 0 or b.std() == 0:
        return 0.0
    return float(max(0.0, np.corrcoef(a, b)[0, 1]))


def metrics(ca, cb, ra, rb, da, db):
    a, b = normalise(ca), normalise(cb)
    ga, gb = cv2.GaussianBlur(a, (0, 0), 2), cv2.GaussianBlur(b, (0, 0), 2)
    structure = float(np.clip(structural_similarity(ga, gb, data_range=255), 0, 1))

    d = cv2.matchShapes(a, b, cv2.CONTOURS_MATCH_I1, 0)
    shape = float(np.exp(-4 * d)) if np.isfinite(d) else 0.0

    sm = lambda v: cv2.GaussianBlur(v.astype(np.float32).reshape(1, -1), (0, 0), 3).ravel()
    strokes = (corr(sm(a.sum(0)), sm(b.sum(0))) + corr(sm(a.sum(1)), sm(b.sum(1)))) / 2

    orb = cv2.ORB_create(500, edgeThreshold=15, patchSize=15)
    ka, xa = orb.detectAndCompute(255 - ga, None)
    kb, xb = orb.detectAndCompute(255 - gb, None)
    keypoints = 0.0
    if xa is not None and xb is not None and len(ka) and len(kb):
        m = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True).match(xa, xb)
        keypoints = min(1.0, sum(x.distance < 50 for x in m) / max(1, min(len(ka), len(kb))) * 2)

    proportion = (1 - abs(ra - rb) / max(ra, rb)) * .5 + (1 - abs(da - db) / max(da, db)) * .5
    return {"structure": structure, "shape": shape, "strokes": strokes,
            "keypoints": keypoints, "proportion": float(proportion)}, a, b


def png64(arr):
    return "data:image/png;base64," + base64.b64encode(cv2.imencode(".png", arr)[1]).decode()


def paint(mask_a, mask_b=None):
    out = np.full((H, W, 3), 255, np.uint8)
    if mask_b is None:
        out[mask_a > 0] = (40, 30, 20)
        return out
    out[mask_a > 0] = (170, 90, 20)    # reference: blue ink (BGR)
    out[mask_b > 0] = (50, 60, 205)    # test: red ink
    out[(mask_a > 0) & (mask_b > 0)] = (30, 25, 20)  # agreement: dark
    return out


@app.route("/")
def index():
    return render_template("index.html")


@app.post("/api/compare")
def compare():
    try:
        fa, fb = request.files.get("reference"), request.files.get("test")
        if not fa or not fb:
            raise ValueError("Add both a reference and a test signature.")
        ca, ra, da = load_ink(fa.read())
        cb, rb, db = load_ink(fb.read())
        m, a, b = metrics(ca, cb, ra, rb, da, db)
        score = round(100 * sum(WEIGHTS[k] * v for k, v in m.items()), 1)
        overlap = float(((a > 0) & (b > 0)).sum() / max(1, ((a > 0) | (b > 0)).sum()))
        return jsonify(score=score, metrics={k: round(v * 100, 1) for k, v in m.items()},
                       weights=WEIGHTS, overlap=round(overlap * 100, 1),
                       images={"reference": png64(paint(a)), "test": png64(paint(b)),
                               "overlay": png64(paint(a, b))})
    except ValueError as e:
        return jsonify(error=str(e)), 400
    except Exception as e:  # noqa
        return jsonify(error=f"Comparison failed: {e}"), 500


if __name__ == "__main__":
    app.run(debug=True)

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import Response
import numpy as np
import cv2
import mediapipe as mp

app = FastAPI()

_face_mesh = None

def get_face_mesh():
    global _face_mesh
    if _face_mesh is None:
        _face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5
        )
    return _face_mesh

LEFT_EYE = [33, 133, 159, 145, 153, 154, 155, 173]
RIGHT_EYE = [362, 263, 386, 374, 380, 381, 382, 398]
LIPS = [61, 291, 13, 14, 0, 17, 78, 308]
NOSE = [1, 2, 98, 327, 168, 197, 5, 4]

FACE_OVAL_PATH = [
    10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
    397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
    172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109
]
LEFT_EYE_CONTOUR = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
RIGHT_EYE_CONTOUR = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
LIPS_CONTOUR = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0, 37, 39, 40, 185]
LEFT_EYEBROW = [70, 63, 105, 66, 107, 55, 65, 52, 53, 46]
RIGHT_EYEBROW = [300, 293, 334, 296, 336, 285, 295, 282, 283, 276]

def landmarks_to_points(landmarks, w, h, idxs):
    pts = []
    for i in idxs:
        lm = landmarks[i]
        pts.append((int(lm.x * w), int(lm.y * h)))
    return np.array(pts, dtype=np.int32)

def center_radius(pts):
    x, y, ww, hh = cv2.boundingRect(pts)
    cx = x + ww / 2.0
    cy = y + hh / 2.0
    r = max(ww, hh) * 0.55
    return cx, cy, r

def create_skin_mask(landmarks, w, h):
    face_pts = landmarks_to_points(landmarks, w, h, FACE_OVAL_PATH)
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillConvexPoly(mask, face_pts, 255)

    for contour in [LEFT_EYE_CONTOUR, RIGHT_EYE_CONTOUR, LIPS_CONTOUR, LEFT_EYEBROW, RIGHT_EYEBROW]:
        pts = landmarks_to_points(landmarks, w, h, contour)
        cv2.fillPoly(mask, [pts], 0)

    mask = cv2.GaussianBlur(mask, (21, 21), 10)
    return mask.astype(np.float32) / 255.0

def blend_with_mask(original, processed, mask):
    mask3 = np.stack([mask] * 3, axis=-1)
    return (original * (1 - mask3) + processed * mask3).astype(np.uint8)

def bulge_pinch(img, cx, cy, radius, strength):
    h, w = img.shape[:2]
    x0 = int(max(cx - radius, 0))
    y0 = int(max(cy - radius, 0))
    x1 = int(min(cx + radius, w - 1))
    y1 = int(min(cy + radius, h - 1))

    roi = img[y0:y1, x0:x1].copy()
    rh, rw = roi.shape[:2]

    yy, xx = np.mgrid[0:rh, 0:rw].astype(np.float32)
    dx = xx - (cx - x0)
    dy = yy - (cy - y0)
    dist = np.sqrt(dx * dx + dy * dy)

    r = float(radius)
    mask = dist < r

    d = np.zeros_like(dist)
    d[mask] = dist[mask] / r

    k = abs(strength)
    if strength >= 0:
        factor = 1 + k * (1 - d * d)
    else:
        factor = 1 - k * (1 - d * d)

    nx = (cx - x0) + dx * factor
    ny = (cy - y0) + dy * factor

    nx[~mask] = xx[~mask]
    ny[~mask] = yy[~mask]

    warped = cv2.remap(
        roi,
        nx,
        ny,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101
    )

    out = img.copy()
    out[y0:y1, x0:x1] = warped
    return out

def apply_feature(img_bgr, face_landmarks, feature, amount):
    h, w = img_bgr.shape[:2]
    strength = max(-0.35, min(0.35, amount / 100.0))

    if feature == "eyes":
        ptsL = landmarks_to_points(face_landmarks, w, h, LEFT_EYE)
        ptsR = landmarks_to_points(face_landmarks, w, h, RIGHT_EYE)
        cxL, cyL, rL = center_radius(ptsL)
        cxR, cyR, rR = center_radius(ptsR)
        img_bgr = bulge_pinch(img_bgr, cxL, cyL, rL, strength)
        img_bgr = bulge_pinch(img_bgr, cxR, cyR, rR, strength)
        return img_bgr

    if feature == "lips":
        pts = landmarks_to_points(face_landmarks, w, h, LIPS)
        cx, cy, r = center_radius(pts)
        return bulge_pinch(img_bgr, cx, cy, r, strength)

    if feature == "nose":
        pts = landmarks_to_points(face_landmarks, w, h, NOSE)
        cx, cy, r = center_radius(pts)
        return bulge_pinch(img_bgr, cx, cy, r, strength)

    raise ValueError("Unknown feature")

# --- Skin effects ---

def skin_smooth(img, mask):
    filtered = cv2.bilateralFilter(img, d=9, sigmaColor=75, sigmaSpace=75)
    return blend_with_mask(img, filtered, mask)

def skin_clear(img, mask):
    filtered = cv2.bilateralFilter(img, d=7, sigmaColor=50, sigmaSpace=50)
    lab = cv2.cvtColor(filtered, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
    l = clahe.apply(l)
    merged = cv2.merge([l, a, b])
    result = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)
    return blend_with_mask(img, result, mask)

def skin_glow(img, mask):
    blurred = cv2.GaussianBlur(img, (0, 0), 25)
    img_f = img.astype(np.float32)
    blur_f = blurred.astype(np.float32)
    screen = 255 - ((255 - img_f) * (255 - blur_f) / 255.0)
    glowed = np.clip(img_f * 0.65 + screen * 0.35, 0, 255).astype(np.uint8)
    lab = cv2.cvtColor(glowed, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = np.clip(l.astype(np.int16) + 10, 0, 255).astype(np.uint8)
    glowed = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)
    return blend_with_mask(img, glowed, mask)

def skin_tan(img, mask):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = np.clip(l.astype(np.int16) - 15, 0, 255).astype(np.uint8)
    a = np.clip(a.astype(np.int16) + 8, 0, 255).astype(np.uint8)
    b = np.clip(b.astype(np.int16) + 10, 0, 255).astype(np.uint8)
    result = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)
    return blend_with_mask(img, result, mask)

def skin_porcelain(img, mask):
    filtered = cv2.bilateralFilter(img, d=15, sigmaColor=100, sigmaSpace=100)
    lab = cv2.cvtColor(filtered, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = np.clip(l.astype(np.int16) + 20, 0, 255).astype(np.uint8)
    a = np.clip(a.astype(np.int16), 124, 132).astype(np.uint8)
    result = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)
    return blend_with_mask(img, result, mask)

def skin_acne(img, mask):
    pass1 = cv2.bilateralFilter(img, d=12, sigmaColor=80, sigmaSpace=80)
    pass2 = cv2.bilateralFilter(pass1, d=9, sigmaColor=60, sigmaSpace=60)
    return blend_with_mask(img, pass2, mask)

SKIN_EFFECTS = {
    "smooth": skin_smooth,
    "clear": skin_clear,
    "glow": skin_glow,
    "tan": skin_tan,
    "porcelain": skin_porcelain,
    "acne": skin_acne,
}

def apply_skin_effect(img_bgr, face_landmarks, effect_name):
    h, w = img_bgr.shape[:2]
    mask = create_skin_mask(face_landmarks, w, h)
    fn = SKIN_EFFECTS[effect_name]
    return fn(img_bgr, mask)

@app.post("/resize")
async def resize(
    image: UploadFile = File(...),
    feature: str = Form(...),
    amount: int = Form(...)
):
    if feature not in ["eyes", "nose", "lips"]:
        raise HTTPException(400, "feature must be one of: eyes, nose, lips")
    if amount < -50 or amount > 50:
        raise HTTPException(400, "amount must be between -50 and 50")

    data = await image.read()
    npimg = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(400, "Invalid image")

    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    res = get_face_mesh().process(rgb)
    if not res.multi_face_landmarks:
        raise HTTPException(422, "No face detected")

    face = res.multi_face_landmarks[0].landmark
    out = apply_feature(img, face, feature, amount)

    ok, buf = cv2.imencode(".jpg", out, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
    if not ok:
        raise HTTPException(500, "Encode failed")

    return Response(content=buf.tobytes(), media_type="image/jpeg")

@app.post("/skin")
async def skin(
    image: UploadFile = File(...),
    effect: str = Form(...)
):
    if effect not in SKIN_EFFECTS:
        raise HTTPException(400, f"effect must be one of: {', '.join(SKIN_EFFECTS.keys())}")

    data = await image.read()
    npimg = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(400, "Invalid image")

    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    res = get_face_mesh().process(rgb)
    if not res.multi_face_landmarks:
        raise HTTPException(422, "No face detected")

    face = res.multi_face_landmarks[0].landmark
    out = apply_skin_effect(img, face, effect)

    ok, buf = cv2.imencode(".jpg", out, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
    if not ok:
        raise HTTPException(500, "Encode failed")

    return Response(content=buf.tobytes(), media_type="image/jpeg")
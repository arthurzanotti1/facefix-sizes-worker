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

UPPER_LIP_OUTER = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 308, 415, 310, 311, 312, 13, 82, 81, 80, 191, 78]
LOWER_LIP_OUTER = [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 291, 375, 321, 405, 314, 17, 84, 181, 91, 146, 61]
LEFT_EYE_UPPER = [246, 161, 160, 159, 158, 157, 173, 133, 155, 154, 153, 145, 144, 163, 7, 33]
RIGHT_EYE_UPPER = [398, 384, 385, 386, 387, 388, 466, 263, 249, 390, 373, 374, 380, 381, 382, 362]

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
    filtered = cv2.bilateralFilter(img, d=11, sigmaColor=80, sigmaSpace=80)
    lab = cv2.cvtColor(filtered, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = np.clip(l.astype(np.int16) + 12, 0, 255).astype(np.uint8)
    a = np.clip(a.astype(np.float32) * 0.85 + 128 * 0.15, 0, 255).astype(np.uint8)
    b = np.clip(b.astype(np.float32) * 0.9 + 128 * 0.1, 0, 255).astype(np.uint8)
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

# --- Makeup effects ---

def create_lips_mask(landmarks, w, h, feather=15):
    upper = landmarks_to_points(landmarks, w, h, UPPER_LIP_OUTER)
    lower = landmarks_to_points(landmarks, w, h, LOWER_LIP_OUTER)
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [upper], 255)
    cv2.fillPoly(mask, [lower], 255)
    ksize = feather * 2 + 1
    mask = cv2.GaussianBlur(mask, (ksize, ksize), feather // 2)
    return mask.astype(np.float32) / 255.0

def create_cheek_mask(landmarks, w, h):
    mask = np.zeros((h, w), dtype=np.uint8)
    nose_tip = landmarks[1]
    left_cheek = landmarks[234]
    right_cheek = landmarks[454]
    nose_x, nose_y = int(nose_tip.x * w), int(nose_tip.y * h)
    lc_x, lc_y = int(left_cheek.x * w), int(left_cheek.y * h)
    rc_x, rc_y = int(right_cheek.x * w), int(right_cheek.y * h)
    radius = int(abs(nose_x - lc_x) * 0.35)
    cl_x = (nose_x + lc_x) // 2
    cl_y = (nose_y + lc_y) // 2 + radius // 3
    cr_x = (nose_x + rc_x) // 2
    cr_y = (nose_y + rc_y) // 2 + radius // 3
    cv2.ellipse(mask, (cl_x, cl_y), (radius, int(radius * 0.7)), 0, 0, 360, 255, -1)
    cv2.ellipse(mask, (cr_x, cr_y), (radius, int(radius * 0.7)), 0, 0, 360, 255, -1)
    mask = cv2.GaussianBlur(mask, (51, 51), 20)
    return mask.astype(np.float32) / 255.0

def create_eye_shadow_mask(landmarks, w, h):
    mask = np.zeros((h, w), dtype=np.uint8)
    for contour, brow in [(LEFT_EYE_UPPER, LEFT_EYEBROW), (RIGHT_EYE_UPPER, RIGHT_EYEBROW)]:
        eye_pts = landmarks_to_points(landmarks, w, h, contour)
        brow_pts = landmarks_to_points(landmarks, w, h, brow)
        region = np.concatenate([eye_pts, brow_pts[::-1]])
        cv2.fillPoly(mask, [region], 255)
    mask = cv2.GaussianBlur(mask, (31, 31), 12)
    return mask.astype(np.float32) / 255.0

def create_contour_mask(landmarks, w, h):
    face_pts = landmarks_to_points(landmarks, w, h, FACE_OVAL_PATH)
    outer = np.zeros((h, w), dtype=np.uint8)
    cv2.fillConvexPoly(outer, face_pts, 255)
    inner = np.zeros((h, w), dtype=np.uint8)
    cx, cy = np.mean(face_pts, axis=0).astype(int)
    shrunk = ((face_pts - [cx, cy]) * 0.7 + [cx, cy]).astype(np.int32)
    cv2.fillConvexPoly(inner, shrunk, 255)
    band = cv2.subtract(outer, inner)
    band = cv2.GaussianBlur(band, (41, 41), 18)
    return band.astype(np.float32) / 255.0

def create_highlight_mask(landmarks, w, h):
    mask = np.zeros((h, w), dtype=np.uint8)
    forehead = landmarks[10]
    nose_bridge = landmarks[6]
    chin = landmarks[152]
    fx, fy = int(forehead.x * w), int(forehead.y * h)
    nx, ny = int(nose_bridge.x * w), int(nose_bridge.y * h)
    cx, cy = int(chin.x * w), int(chin.y * h)
    face_w = int(abs(landmarks[234].x - landmarks[454].x) * w)
    r = face_w // 5
    cv2.ellipse(mask, (fx, fy), (r, int(r * 0.6)), 0, 0, 360, 255, -1)
    cv2.ellipse(mask, (nx, ny), (int(r * 0.4), int(r * 1.2)), 0, 0, 360, 255, -1)
    cv2.ellipse(mask, (cx, cy), (int(r * 0.6), int(r * 0.4)), 0, 0, 360, 255, -1)
    mask = cv2.GaussianBlur(mask, (51, 51), 22)
    return mask.astype(np.float32) / 255.0

def makeup_natural(img, landmarks, w, h):
    skin_mask = create_skin_mask(landmarks, w, h)
    smoothed = cv2.bilateralFilter(img, d=7, sigmaColor=50, sigmaSpace=50)
    result = blend_with_mask(img, smoothed, skin_mask * 0.5)
    lips_mask = create_lips_mask(landmarks, w, h, feather=11)
    lip_tint = result.copy()
    lip_tint[:, :, 1] = np.clip(lip_tint[:, :, 1].astype(np.int16) - 10, 0, 255).astype(np.uint8)
    lip_tint[:, :, 2] = np.clip(lip_tint[:, :, 2].astype(np.int16) + 15, 0, 255).astype(np.uint8)
    result = blend_with_mask(result, lip_tint, lips_mask * 0.4)
    return result

def makeup_glam(img, landmarks, w, h):
    skin_mask = create_skin_mask(landmarks, w, h)
    smoothed = cv2.bilateralFilter(img, d=9, sigmaColor=70, sigmaSpace=70)
    result = blend_with_mask(img, smoothed, skin_mask * 0.6)
    lab = cv2.cvtColor(result, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = np.clip(l.astype(np.int16) + 8, 0, 255).astype(np.uint8)
    bright = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)
    result = blend_with_mask(result, bright, skin_mask * 0.5)
    lips_mask = create_lips_mask(landmarks, w, h)
    lip_color = result.copy()
    lip_color[:, :, 0] = np.clip(lip_color[:, :, 0].astype(np.int16) - 20, 0, 255).astype(np.uint8)
    lip_color[:, :, 1] = np.clip(lip_color[:, :, 1].astype(np.int16) - 15, 0, 255).astype(np.uint8)
    lip_color[:, :, 2] = np.clip(lip_color[:, :, 2].astype(np.int16) + 30, 0, 255).astype(np.uint8)
    result = blend_with_mask(result, lip_color, lips_mask * 0.5)
    eye_mask = create_eye_shadow_mask(landmarks, w, h)
    eye_dark = result.copy()
    eye_lab = cv2.cvtColor(eye_dark, cv2.COLOR_BGR2LAB)
    el, ea, eb = cv2.split(eye_lab)
    el = np.clip(el.astype(np.int16) - 15, 0, 255).astype(np.uint8)
    eye_dark = cv2.cvtColor(cv2.merge([el, ea, eb]), cv2.COLOR_LAB2BGR)
    result = blend_with_mask(result, eye_dark, eye_mask * 0.3)
    return result

def makeup_smoky(img, landmarks, w, h):
    eye_mask = create_eye_shadow_mask(landmarks, w, h)
    shadow = img.copy()
    shadow_lab = cv2.cvtColor(shadow, cv2.COLOR_BGR2LAB)
    sl, sa, sb = cv2.split(shadow_lab)
    sl = np.clip(sl.astype(np.int16) - 40, 0, 255).astype(np.uint8)
    shadow = cv2.cvtColor(cv2.merge([sl, sa, sb]), cv2.COLOR_LAB2BGR)
    shadow[:, :, 0] = np.clip(shadow[:, :, 0].astype(np.int16) + 10, 0, 255).astype(np.uint8)
    result = blend_with_mask(img, shadow, eye_mask * 0.55)
    skin_mask = create_skin_mask(landmarks, w, h)
    smoothed = cv2.bilateralFilter(result, d=5, sigmaColor=40, sigmaSpace=40)
    result = blend_with_mask(result, smoothed, skin_mask * 0.3)
    return result

def makeup_blush(img, landmarks, w, h):
    cheek_mask = create_cheek_mask(landmarks, w, h)
    blush_overlay = img.copy()
    blush_overlay[:, :, 0] = np.clip(blush_overlay[:, :, 0].astype(np.int16) - 15, 0, 255).astype(np.uint8)
    blush_overlay[:, :, 1] = np.clip(blush_overlay[:, :, 1].astype(np.int16) - 10, 0, 255).astype(np.uint8)
    blush_overlay[:, :, 2] = np.clip(blush_overlay[:, :, 2].astype(np.int16) + 25, 0, 255).astype(np.uint8)
    result = blend_with_mask(img, blush_overlay, cheek_mask * 0.45)
    skin_mask = create_skin_mask(landmarks, w, h)
    smoothed = cv2.bilateralFilter(result, d=5, sigmaColor=40, sigmaSpace=40)
    result = blend_with_mask(result, smoothed, skin_mask * 0.25)
    return result

def makeup_lipstick(img, landmarks, w, h):
    lips_mask = create_lips_mask(landmarks, w, h, feather=9)
    lip_color = img.copy()
    lip_color[:, :, 0] = np.clip(lip_color[:, :, 0].astype(np.int16) - 40, 0, 255).astype(np.uint8)
    lip_color[:, :, 1] = np.clip(lip_color[:, :, 1].astype(np.int16) - 30, 0, 255).astype(np.uint8)
    lip_color[:, :, 2] = np.clip(lip_color[:, :, 2].astype(np.int16) + 50, 0, 255).astype(np.uint8)
    result = blend_with_mask(img, lip_color, lips_mask * 0.6)
    return result

def makeup_contour(img, landmarks, w, h):
    contour_mask = create_contour_mask(landmarks, w, h)
    darkened = img.copy()
    dark_lab = cv2.cvtColor(darkened, cv2.COLOR_BGR2LAB)
    dl, da, db = cv2.split(dark_lab)
    dl = np.clip(dl.astype(np.int16) - 20, 0, 255).astype(np.uint8)
    da = np.clip(da.astype(np.int16) + 4, 0, 255).astype(np.uint8)
    darkened = cv2.cvtColor(cv2.merge([dl, da, db]), cv2.COLOR_LAB2BGR)
    result = blend_with_mask(img, darkened, contour_mask * 0.5)
    highlight_mask = create_highlight_mask(landmarks, w, h)
    bright = result.copy()
    bright_lab = cv2.cvtColor(bright, cv2.COLOR_BGR2LAB)
    bl, ba, bb = cv2.split(bright_lab)
    bl = np.clip(bl.astype(np.int16) + 15, 0, 255).astype(np.uint8)
    bright = cv2.cvtColor(cv2.merge([bl, ba, bb]), cv2.COLOR_LAB2BGR)
    result = blend_with_mask(result, bright, highlight_mask * 0.4)
    skin_mask = create_skin_mask(landmarks, w, h)
    smoothed = cv2.bilateralFilter(result, d=5, sigmaColor=40, sigmaSpace=40)
    result = blend_with_mask(result, smoothed, skin_mask * 0.25)
    return result

MAKEUP_EFFECTS = {
    "natural": makeup_natural,
    "glam": makeup_glam,
    "smoky": makeup_smoky,
    "blush": makeup_blush,
    "lipstick": makeup_lipstick,
    "contour": makeup_contour,
}

def apply_makeup_effect(img_bgr, face_landmarks, effect_name):
    h, w = img_bgr.shape[:2]
    fn = MAKEUP_EFFECTS[effect_name]
    return fn(img_bgr, face_landmarks, w, h)

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

@app.post("/makeup")
async def makeup(
    image: UploadFile = File(...),
    effect: str = Form(...)
):
    if effect not in MAKEUP_EFFECTS:
        raise HTTPException(400, f"effect must be one of: {', '.join(MAKEUP_EFFECTS.keys())}")

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
    out = apply_makeup_effect(img, face, effect)

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
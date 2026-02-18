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

FACE_OVAL = [
    10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
    397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
    172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109
]
LIPS_OUTER = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0, 37, 39, 40, 185]
LIPS_INNER = [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 415, 310, 311, 312, 13, 82, 81, 80, 191]
LEFT_EYE_FULL = [
    33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158,
    159, 160, 161, 246
]
RIGHT_EYE_FULL = [
    362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387,
    386, 385, 384, 398
]
LEFT_EYEBROW = [70, 63, 105, 66, 107, 55, 65, 52, 53, 46]
RIGHT_EYEBROW = [300, 293, 334, 296, 336, 285, 295, 282, 283, 276]
LEFT_CHEEK = [117, 118, 101, 36, 205, 187, 123, 116, 111, 100]
RIGHT_CHEEK = [346, 347, 330, 266, 425, 411, 352, 345, 340, 329]
NOSE_BRIDGE_LEFT = [193, 122, 196, 3, 51, 45]
NOSE_BRIDGE_RIGHT = [417, 351, 419, 248, 281, 275]
FOREHEAD_CENTER = [10, 338, 297, 332, 284, 251, 21, 54, 103, 67, 109]
JAWLINE = [
    172, 58, 132, 93, 234, 127, 162, 21,
    389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152,
    148, 176, 149, 150, 136
]

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

def create_region_mask(landmarks, w, h, indices, feather=11):
    pts = landmarks_to_points(landmarks, w, h, indices)
    hull = cv2.convexHull(pts)
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillConvexPoly(mask, hull, 255)
    if feather > 0:
        k = feather | 1
        mask = cv2.GaussianBlur(mask, (k, k), 0)
    return mask.astype(np.float32) / 255.0


def create_lips_mask(landmarks, w, h, feather=7):
    outer = landmarks_to_points(landmarks, w, h, LIPS_OUTER)
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [outer], 255)
    if feather > 0:
        k = feather | 1
        mask = cv2.GaussianBlur(mask, (k, k), 0)
    return mask.astype(np.float32) / 255.0


def create_eye_mask(landmarks, w, h, eye_indices, expand=0, feather=9):
    pts = landmarks_to_points(landmarks, w, h, eye_indices)
    if expand > 0:
        pts_expanded = pts.copy()
        cy = np.mean(pts[:, 1])
        for i in range(len(pts_expanded)):
            if pts_expanded[i][1] < cy:
                pts_expanded[i][1] -= expand
        pts = pts_expanded
    hull = cv2.convexHull(pts)
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillConvexPoly(mask, hull, 255)
    if feather > 0:
        k = feather | 1
        mask = cv2.GaussianBlur(mask, (k, k), 0)
    return mask.astype(np.float32) / 255.0


def alpha_blend(base, overlay, mask):
    mask3 = np.stack([mask] * 3, axis=-1)
    return (base.astype(np.float32) * (1 - mask3) + overlay.astype(np.float32) * mask3).clip(0, 255).astype(np.uint8)


def color_overlay(img, mask, color_bgr, opacity):
    overlay = np.full_like(img, color_bgr, dtype=np.uint8)
    blended = alpha_blend(img, overlay, mask * opacity)
    return blended


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


SKIN_EFFECTS = ["smooth", "clear", "glow", "tan", "porcelain", "acne"]


def make_skin_mask(landmarks, w, h):
    face = create_region_mask(landmarks, w, h, FACE_OVAL, feather=15)
    leye = create_eye_mask(landmarks, w, h, LEFT_EYE_FULL, feather=7)
    reye = create_eye_mask(landmarks, w, h, RIGHT_EYE_FULL, feather=7)
    lips = create_lips_mask(landmarks, w, h, feather=7)
    mask = face - leye - reye - lips
    return np.clip(mask, 0, 1)


def apply_skin(img, landmarks, effect):
    h, w = img.shape[:2]
    mask = make_skin_mask(landmarks, w, h)

    if effect == "smooth":
        smoothed = cv2.bilateralFilter(img, 9, 75, 75)
        return alpha_blend(img, smoothed, mask * 0.6)

    if effect == "clear":
        p1 = cv2.bilateralFilter(img, 9, 75, 75)
        p2 = cv2.bilateralFilter(p1, 9, 75, 75)
        return alpha_blend(img, p2, mask * 0.75)

    if effect == "glow":
        blurred = cv2.GaussianBlur(img, (0, 0), 15)
        screen = 255 - ((255 - img.astype(np.uint16)) * (255 - blurred.astype(np.uint16)) // 255).astype(np.uint8)
        bright = cv2.convertScaleAbs(screen, alpha=1.08, beta=0)
        return alpha_blend(img, bright, mask * 0.5)

    if effect == "tan":
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 0] = np.clip(hsv[:, :, 0] - 5, 0, 179)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 1.2, 0, 255)
        hsv[:, :, 2] = np.clip(hsv[:, :, 2] * 0.92, 0, 255)
        tanned = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
        return alpha_blend(img, tanned, mask * 0.65)

    if effect == "porcelain":
        smoothed = cv2.bilateralFilter(img, 9, 75, 75)
        hsv = cv2.cvtColor(smoothed, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 0.65, 0, 255)
        hsv[:, :, 2] = np.clip(hsv[:, :, 2] * 1.15, 0, 255)
        porcelain = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
        return alpha_blend(img, porcelain, mask * 0.6)

    if effect == "acne":
        med = cv2.medianBlur(img, 7)
        smoothed = cv2.bilateralFilter(med, 9, 75, 75)
        return alpha_blend(img, smoothed, mask * 0.8)

    raise ValueError("Unknown skin effect")


@app.post("/skin")
async def skin(
    image: UploadFile = File(...),
    effect: str = Form(...)
):
    if effect not in SKIN_EFFECTS:
        raise HTTPException(400, f"effect must be one of: {', '.join(SKIN_EFFECTS)}")

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
    out = apply_skin(img, face, effect)

    ok, buf = cv2.imencode(".jpg", out, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
    if not ok:
        raise HTTPException(500, "Encode failed")

    return Response(content=buf.tobytes(), media_type="image/jpeg")


MAKEUP_EFFECTS = ["natural", "glam", "smoky", "blush", "lipstick", "contour"]


def apply_makeup(img, landmarks, effect):
    h, w = img.shape[:2]

    if effect == "natural":
        lips = create_lips_mask(landmarks, w, h, feather=7)
        out = color_overlay(img, lips, (80, 100, 180), 0.20)
        lcheek = create_region_mask(landmarks, w, h, LEFT_CHEEK, feather=21)
        rcheek = create_region_mask(landmarks, w, h, RIGHT_CHEEK, feather=21)
        cheeks = np.clip(lcheek + rcheek, 0, 1)
        out = color_overlay(out, cheeks, (120, 130, 190), 0.12)
        leye = create_eye_mask(landmarks, w, h, LEFT_EYE_FULL, feather=9)
        reye = create_eye_mask(landmarks, w, h, RIGHT_EYE_FULL, feather=9)
        eyes = np.clip(leye + reye, 0, 1)
        hsv = cv2.cvtColor(out, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 1.15, 0, 255)
        hsv[:, :, 2] = np.clip(hsv[:, :, 2] * 0.92, 0, 255)
        contrast_eyes = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
        return alpha_blend(out, contrast_eyes, eyes * 0.30)

    if effect == "glam":
        lips = create_lips_mask(landmarks, w, h, feather=5)
        out = color_overlay(img, lips, (60, 50, 170), 0.40)
        leye = create_eye_mask(landmarks, w, h, LEFT_EYE_FULL, expand=15, feather=15)
        reye = create_eye_mask(landmarks, w, h, RIGHT_EYE_FULL, expand=15, feather=15)
        eyes = np.clip(leye + reye, 0, 1)
        out = color_overlay(out, eyes, (60, 90, 140), 0.20)
        forehead = create_region_mask(landmarks, w, h, FOREHEAD_CENTER, feather=25)
        nose_l = create_region_mask(landmarks, w, h, NOSE_BRIDGE_LEFT, feather=15)
        nose_r = create_region_mask(landmarks, w, h, NOSE_BRIDGE_RIGHT, feather=15)
        tzone = np.clip(forehead + nose_l + nose_r, 0, 1)
        highlight = cv2.convertScaleAbs(out, alpha=1.12, beta=10)
        return alpha_blend(out, highlight, tzone * 0.35)

    if effect == "smoky":
        leye_wide = create_eye_mask(landmarks, w, h, LEFT_EYE_FULL, expand=12, feather=21)
        reye_wide = create_eye_mask(landmarks, w, h, RIGHT_EYE_FULL, expand=12, feather=21)
        eyes_wide = np.clip(leye_wide + reye_wide, 0, 1)
        out = color_overlay(img, eyes_wide, (40, 35, 35), 0.35)
        leye_tight = create_eye_mask(landmarks, w, h, LEFT_EYE_FULL, expand=3, feather=7)
        reye_tight = create_eye_mask(landmarks, w, h, RIGHT_EYE_FULL, expand=3, feather=7)
        eyes_tight = np.clip(leye_tight + reye_tight, 0, 1)
        return color_overlay(out, eyes_tight, (30, 25, 25), 0.30)

    if effect == "blush":
        lcheek = create_region_mask(landmarks, w, h, LEFT_CHEEK, feather=31)
        rcheek = create_region_mask(landmarks, w, h, RIGHT_CHEEK, feather=31)
        cheeks = np.clip(lcheek + rcheek, 0, 1)
        return color_overlay(img, cheeks, (140, 130, 200), 0.25)

    if effect == "lipstick":
        lips = create_lips_mask(landmarks, w, h, feather=5)
        out = color_overlay(img, lips, (50, 50, 190), 0.45)
        inner = create_lips_mask(landmarks, w, h, feather=3)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        inner_eroded = cv2.erode((inner * 255).astype(np.uint8), kernel, iterations=1).astype(np.float32) / 255.0
        gloss = cv2.convertScaleAbs(out, alpha=1.15, beta=5)
        return alpha_blend(out, gloss, inner_eroded * 0.40)

    if effect == "contour":
        jaw = create_region_mask(landmarks, w, h, JAWLINE, feather=21)
        nose_l = create_region_mask(landmarks, w, h, NOSE_BRIDGE_LEFT, feather=11)
        nose_r = create_region_mask(landmarks, w, h, NOSE_BRIDGE_RIGHT, feather=11)
        darken_mask = np.clip(jaw + nose_l + nose_r, 0, 1)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 1.10, 0, 255)
        hsv[:, :, 2] = np.clip(hsv[:, :, 2] * 0.82, 0, 255)
        darkened = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
        out = alpha_blend(img, darkened, darken_mask * 0.6)
        forehead = create_region_mask(landmarks, w, h, FOREHEAD_CENTER, feather=21)
        nose_bridge = create_region_mask(landmarks, w, h, NOSE, feather=11)
        tzone = np.clip(forehead + nose_bridge, 0, 1)
        hsv2 = cv2.cvtColor(out, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv2[:, :, 2] = np.clip(hsv2[:, :, 2] * 1.12, 0, 255)
        brightened = cv2.cvtColor(hsv2.astype(np.uint8), cv2.COLOR_HSV2BGR)
        return alpha_blend(out, brightened, tzone * 0.5)

    raise ValueError("Unknown makeup effect")


@app.post("/makeup")
async def makeup(
    image: UploadFile = File(...),
    effect: str = Form(...)
):
    if effect not in MAKEUP_EFFECTS:
        raise HTTPException(400, f"effect must be one of: {', '.join(MAKEUP_EFFECTS)}")

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
    out = apply_makeup(img, face, effect)

    ok, buf = cv2.imencode(".jpg", out, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
    if not ok:
        raise HTTPException(500, "Encode failed")

    return Response(content=buf.tobytes(), media_type="image/jpeg")
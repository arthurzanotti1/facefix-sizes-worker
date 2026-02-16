from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import Response
import numpy as np
import cv2
import mediapipe as mp

app = FastAPI()

mp_face_mesh = mp.solutions.face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5
)

LEFT_EYE = [33, 133, 159, 145, 153, 154, 155, 173]
RIGHT_EYE = [362, 263, 386, 374, 380, 381, 382, 398]
LIPS = [61, 291, 13, 14, 0, 17, 78, 308]
NOSE = [1, 2, 98, 327, 168, 197, 5, 4]

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
    res = mp_face_mesh.process(rgb)
    if not res.multi_face_landmarks:
        raise HTTPException(422, "No face detected")

    face = res.multi_face_landmarks[0].landmark
    out = apply_feature(img, face, feature, amount)

    ok, buf = cv2.imencode(".jpg", out, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
    if not ok:
        raise HTTPException(500, "Encode failed")

    return Response(content=buf.tobytes(), media_type="image/jpeg")
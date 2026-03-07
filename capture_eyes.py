import cv2
import mediapipe as mp
import numpy as np
import os

# ================= SETTINGS =================
SAVE_DIR = "dataset"
OPEN_DIR = os.path.join(SAVE_DIR, "open")
CLOSED_DIR = os.path.join(SAVE_DIR, "closed")

MAX_IMAGES_PER_CLASS = 400
IMG_SIZE = 224

OPEN_EAR_THRESHOLD = 0.28
CLOSED_EAR_THRESHOLD = 0.18

# ============================================

os.makedirs(OPEN_DIR, exist_ok=True)
os.makedirs(CLOSED_DIR, exist_ok=True)

open_count = len(os.listdir(OPEN_DIR))
closed_count = len(os.listdir(CLOSED_DIR))

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)

LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

def eye_aspect_ratio(landmarks, eye_indices):
    p = [np.array([landmarks[i].x, landmarks[i].y]) for i in eye_indices]
    vertical1 = np.linalg.norm(p[1] - p[5])
    vertical2 = np.linalg.norm(p[2] - p[4])
    horizontal = np.linalg.norm(p[0] - p[3])
    return (vertical1 + vertical2) / (2.0 * horizontal)

cap = cv2.VideoCapture(0)

print("📸 Eye Capture Started")
print("Keep eyes fully OPEN or fully CLOSED") 
print("Press 'q' to quit")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    if results.multi_face_landmarks:
        for face in results.multi_face_landmarks:

            left_ear = eye_aspect_ratio(face.landmark, LEFT_EYE)
            right_ear = eye_aspect_ratio(face.landmark, RIGHT_EYE)
            ear = (left_ear + right_ear) / 2

            if ear > OPEN_EAR_THRESHOLD:
                label = "open"
            elif ear < CLOSED_EAR_THRESHOLD:
                label = "closed"
            else:
                label = "ignore"

            for eye_indices in [LEFT_EYE, RIGHT_EYE]:
                xs = [int(face.landmark[i].x * w) for i in eye_indices]
                ys = [int(face.landmark[i].y * h) for i in eye_indices]

                min_x = max(min(xs) - 20, 0)
                max_x = min(max(xs) + 20, w)
                min_y = max(min(ys) - 20, 0)
                max_y = min(max(ys) + 20, h)

                eye_img = frame[min_y:max_y, min_x:max_x]
                if eye_img.size == 0:
                    continue

                eye_img = cv2.resize(eye_img, (IMG_SIZE, IMG_SIZE))

                if label == "open" and open_count < MAX_IMAGES_PER_CLASS:
                    cv2.imwrite(f"{OPEN_DIR}/open_{open_count}.jpg", eye_img)
                    open_count += 1

                elif label == "closed" and closed_count < MAX_IMAGES_PER_CLASS:
                    cv2.imwrite(f"{CLOSED_DIR}/closed_{closed_count}.jpg", eye_img)
                    closed_count += 1

            cv2.putText(frame, f"Eye: {label.upper()}", (30, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Eye Capture", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    if open_count >= MAX_IMAGES_PER_CLASS and closed_count >= MAX_IMAGES_PER_CLASS:
        break

cap.release()
cv2.destroyAllWindows()

print("✅ Dataset capture completed")
print(f"Open images: {open_count}")
print(f"Closed images: {closed_count}")

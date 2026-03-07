import cv2
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import load_model
import time

# Load trained model
model = load_model('eye_state_model.h5')
print("Model loaded successfully!")

# MediaPipe face mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False,
                                  max_num_faces=1,
                                  min_detection_confidence=0.5,
                                  min_tracking_confidence=0.5)

# Drawing utils
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Eye landmark indices (left and right)
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

# Start webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open webcam")
    exit()

# Timing counters
closed_start = None
open_start = None
alert_displayed = False

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(frame_rgb)

    eye_state = "Unknown"

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            h, w, _ = frame.shape

            # Function to crop eye
            def crop_eye(landmarks, eye_indices):
                xs = [int(landmarks.landmark[i].x * w) for i in eye_indices]
                ys = [int(landmarks.landmark[i].y * h) for i in eye_indices]
                x1, x2 = max(min(xs)-5, 0), min(max(xs)+5, w)
                y1, y2 = max(min(ys)-5, 0), min(max(ys)+5, h)

                return frame[y1:y2, x1:x2], (x1, y1, x2, y2)

            # Left eye
            left_eye_img, left_coords = crop_eye(face_landmarks, LEFT_EYE)
            # Right eye
            right_eye_img, right_coords = crop_eye(face_landmarks, RIGHT_EYE)

            # Predict function
            def predict_eye(img):
                try:
                    img = cv2.resize(img, (64,64))
                    img = img / 255.0
                    img = np.expand_dims(img, axis=0)
                    pred = model.predict(img)[0][0]
                    return "closed" if pred >= 0.5 else "open"
                except:
                    return "Unknown"

            left_state = predict_eye(left_eye_img)
            right_state = predict_eye(right_eye_img)

            # Decide overall eye state
            if left_state == "Closed" and right_state == "Closed":
                eye_state = "Closed"
            elif left_state == "Open" and right_state == "Open":
                eye_state = "Open"
            else:
                eye_state = "Open"

            # Draw rectangles
            cv2.rectangle(frame, (left_coords[0], left_coords[1]), (left_coords[2], left_coords[3]), (0,255,0), 1)
            cv2.rectangle(frame, (right_coords[0], right_coords[1]), (right_coords[2], right_coords[3]), (0,255,0), 1)

    # Alert logic
    current_time = time.time()
    if eye_state == "Closed":
        if closed_start is None:
            closed_start = current_time
        if current_time - closed_start > 2:
            cv2.putText(frame, "DROWSY!", (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
            alert_displayed = True
    else:
        closed_start = None

    if eye_state == "Open":
        if open_start is None:
            open_start = current_time
        if current_time - open_start > 5:
            cv2.putText(frame, "EYE STRAIN!", (50,100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,255), 2)
    else:
        open_start = None

    # Show eye state
    cv2.putText(frame, f"Eye State: {eye_state}", (50,150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,0), 2)

    # Display the frame
    cv2.imshow("EyeGuard AI", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
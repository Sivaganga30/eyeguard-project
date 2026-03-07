import cv2
import mediapipe as mp
import math
import time
import threading
import winsound

# ---------- Settings ----------
EAR_THRESHOLD = 0.30      # below this = eye closed
DROWSY_CLOSED_SEC = 2.0   # seconds eyes closed to consider drowsy
OPEN_STARE_SEC = 5.0      # seconds eyes open to consider eye strain

# ---------- Mediapipe Face Mesh ----------
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True)

# ---------- Eye Landmarks ----------
LEFT_EYE = [33,160,158,133,153,144]
RIGHT_EYE = [362,385,387,263,373,380]

# ---------- EAR Function ----------
def eye_aspect_ratio(landmarks, eye_points, frame_shape):
    coords = [(landmarks[p].x*frame_shape[1], landmarks[p].y*frame_shape[0]) for p in eye_points]
    A = math.dist(coords[1], coords[5])
    B = math.dist(coords[2], coords[4])
    C = math.dist(coords[0], coords[3])
    return (A+B)/(2*C) if C!=0 else 0

# ---------- Async Beep Function ----------
def beep_async(freq=750, duration=300):
    threading.Thread(target=winsound.Beep, args=(freq,duration), daemon=True).start()

# ---------- State Variables ----------
blink_count = 0
blink_timestamps = []
eyes_were_closed = False
eyes_closed_start = None
eyes_open_start = None

# ---------- Webcam Capture ----------
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # --- EAR Calculation ---
            left_ear = eye_aspect_ratio(face_landmarks.landmark, LEFT_EYE, frame.shape)
            right_ear = eye_aspect_ratio(face_landmarks.landmark, RIGHT_EYE, frame.shape)
            avg_ear = (left_ear + right_ear)/2
            cv2.putText(frame, f"EAR: {avg_ear:.2f}", (30,30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)

            # --- Blink & Drowsiness Detection ---
            current_time = time.time()

            if avg_ear < EAR_THRESHOLD:
                if eyes_closed_start is None:
                    eyes_closed_start = current_time
                if current_time - eyes_closed_start >= DROWSY_CLOSED_SEC:
                    cv2.putText(frame, "DROWSY!", (30,70),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255), 2)
                    beep_async(800,300)
                eyes_were_closed = True
                eyes_open_start = None
            else:
                # Blink Detection
                if eyes_were_closed:
                    blink_count += 1
                    blink_timestamps.append(current_time)
                    eyes_were_closed = False
                # Prolonged stare (eye strain)
                if eyes_open_start is None:
                    eyes_open_start = current_time
                elif current_time - eyes_open_start >= OPEN_STARE_SEC:
                    cv2.putText(frame, "EYE STRAIN!", (30,100),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,165,255), 2)
                    beep_async(600,250)
                eyes_closed_start = None

            # --- Display Blink Count ---
            cv2.putText(frame, f"Blink Count: {blink_count}", (30,150),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,0,0),2)

    cv2.imshow("EyeGuard Test", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
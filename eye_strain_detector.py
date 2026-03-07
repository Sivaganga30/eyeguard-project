import screen_brightness_control as sbc
import cv2
import mediapipe as mp
import math
import time
import winsound
import tkinter as tk
from threading import Thread
import numpy as np
import pyttsx3


# ================= SETTINGS =================
EAR_OPEN_THRESHOLD = 0.28
EAR_CLOSED_THRESHOLD = 0.20

STRAIN_TIME = 5
DROWSY_TIME = 2

MIN_BRIGHTNESS = 30
MAX_BRIGHTNESS = 90

BRIGHTNESS_UPDATE_INTERVAL = 3
BRIGHTNESS_CHANGE_THRESHOLD = 10

RULE_TIME = 30  # change to 1200 for real 20 minutes
# ===========================================


# ---------- VOICE ----------
def speak(text):
    engine = pyttsx3.init()
    engine.setProperty('rate',150)
    engine.say(text)
    engine.runAndWait()
    engine.stop()


# ---------- MEDIAPIPE ----------
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)

LEFT_EYE = [33,160,158,133,153,144]
RIGHT_EYE = [362,385,387,263,373,380]


# ---------- EAR ----------
def eye_aspect_ratio(landmarks, eye):

    p = [(landmarks[i].x, landmarks[i].y) for i in eye]

    A = math.dist(p[1], p[5])
    B = math.dist(p[2], p[4])
    C = math.dist(p[0], p[3])

    return (A + B) / (2.0 * C)


# ---------- AUTO BRIGHTNESS ----------
last_brightness_update = 0
current_system_brightness = None

def auto_adjust_brightness(frame):

    global last_brightness_update, current_system_brightness

    if time.time() - last_brightness_update < BRIGHTNESS_UPDATE_INTERVAL:
        return current_system_brightness

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    avg_light = np.mean(gray)

    new_brightness = int(np.interp(
        avg_light,
        [40,200],
        [MIN_BRIGHTNESS,MAX_BRIGHTNESS]
    ))

    if (
        current_system_brightness is None or
        abs(new_brightness - current_system_brightness) >= BRIGHTNESS_CHANGE_THRESHOLD
    ):
        try:
            sbc.set_brightness(new_brightness)
            current_system_brightness = new_brightness
            last_brightness_update = time.time()
        except:
            pass

    return current_system_brightness


# ---------- POPUP ----------
def show_break_popup():

    root = tk.Tk()
    root.title("EyeGuard Alert")
    root.attributes("-topmost", True)
    root.geometry("380x160")

    tk.Label(
        root,
        text="⚠ TAKE A BREAK\n\nYou have been staring too long",
        font=("Arial",14),
        fg="red"
    ).pack(expand=True,pady=20)

    tk.Button(root,text="OK",command=root.destroy).pack()

    root.mainloop()


# ---------- EYE EXERCISE ----------
def show_eye_exercise():

    root = tk.Tk()
    root.title("Eye Exercise Suggestion")
    root.attributes("-topmost", True)
    root.geometry("420x220")

    message = """
EYE RELAXATION EXERCISES

Blink rapidly for 10 seconds
Look left and right slowly
Look at distant object for 20 seconds
Close eyes and relax
"""

    tk.Label(root,text=message,font=("Arial",12)).pack(pady=20)

    tk.Button(root,text="OK",command=root.destroy).pack()

    root.mainloop()


# ---------- MAIN ----------
cap = cv2.VideoCapture(0)

open_start = None
closed_start = None

strain_alert = False
drowsy_alert = False

blink_count = 0
blink_start_time = time.time()
eye_closed = False

rule_start_time = time.time()

looking_at_screen = False

print("EyeGuard Running")


while cap.isOpened():

    ret, frame = cap.read()

    if not ret:
        break

    brightness = auto_adjust_brightness(frame)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    label = "Detecting..."
    color = (255,255,0)

    if results.multi_face_landmarks:

        face = results.multi_face_landmarks[0]

        # ---------- FACE ORIENTATION CHECK ----------
        nose = face.landmark[1]
        left_eye_corner = face.landmark[33]
        right_eye_corner = face.landmark[263]

        face_center = (left_eye_corner.x + right_eye_corner.x) / 2

        if abs(nose.x - face_center) > 0.05:

            label = "LOOK AT SCREEN"
            color = (0,255,255)

            looking_at_screen = False

            open_start = None
            closed_start = None
            strain_alert = False
            drowsy_alert = False

        else:

            looking_at_screen = True

            ear = (
                eye_aspect_ratio(face.landmark,LEFT_EYE) +
                eye_aspect_ratio(face.landmark,RIGHT_EYE)
            ) / 2


            # ---------- BLINK DETECTION ----------
            if ear < EAR_CLOSED_THRESHOLD:

                eye_closed = True

            elif ear > EAR_OPEN_THRESHOLD:

                if eye_closed:
                    blink_count += 1
                    eye_closed = False


            # ---------- DROWSINESS ----------
            if ear < EAR_CLOSED_THRESHOLD:

                if closed_start is None:
                    closed_start = time.time()

                closed_time = time.time() - closed_start

                if closed_time > DROWSY_TIME:

                    label = "DROWSY"
                    color = (0,0,255)

                    if not drowsy_alert:

                        winsound.Beep(1500,500)

                        Thread(target=speak,
                               args=("You look drowsy. Please wake up.",),
                               daemon=True).start()

                        drowsy_alert = True

                else:
                    label = "EYES CLOSED"
                    color = (0,0,255)

                open_start = None
                strain_alert = False


            # ---------- EYE STRAIN ----------
            elif ear > EAR_OPEN_THRESHOLD:

                closed_start = None
                drowsy_alert = False

                if open_start is None:
                    open_start = time.time()

                elapsed = time.time() - open_start

                if elapsed >= STRAIN_TIME:

                    label = "EYE STRAIN DETECTED"
                    color = (0,0,255)

                    if not strain_alert:

                        winsound.Beep(1000,500)

                        Thread(target=speak,
                               args=("Eye strain detected. Please take a break.",),
                               daemon=True).start()

                        Thread(target=show_break_popup,daemon=True).start()
                        Thread(target=show_eye_exercise,daemon=True).start()

                        strain_alert = True

                else:
                    label = "EYES OPEN"
                    color = (0,255,0)

                    strain_alert = False


            # ---------- BLINK RATE ----------
            elapsed_time = time.time() - blink_start_time

            if elapsed_time >= 60:

                blink_rate = blink_count
                blink_count = 0
                blink_start_time = time.time()

                if blink_rate < 10:

                    winsound.Beep(1200,500)

                    Thread(target=speak,
                           args=("Your blink rate is low. Please blink more often.",),
                           daemon=True).start()


    else:

        label = "FACE NOT DETECTED"
        color = (200,200,200)

        looking_at_screen = False

        open_start = None
        closed_start = None
        strain_alert = False
        drowsy_alert = False


    # ---------- 20-20-20 RULE ----------
    if looking_at_screen:

        rule_elapsed = time.time() - rule_start_time

        if rule_elapsed >= RULE_TIME:

            winsound.Beep(1200,500)

            Thread(target=speak,
                   args=("Please follow the twenty twenty twenty rule. Look away from the screen for twenty seconds.",),
                   daemon=True).start()

            rule_start_time = time.time()

    else:
        rule_start_time = time.time()


    # ---------- DISPLAY ----------
    cv2.putText(frame,label,(30,40),
                cv2.FONT_HERSHEY_SIMPLEX,1,color,2)

    cv2.putText(frame,
                f"Blinks: {blink_count}",
                (30,120),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255,255,255),
                2)

    if brightness is not None:

        cv2.putText(frame,
                    f"Auto Brightness: {brightness}%",
                    (30,80),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255,255,255),
                    2)

    cv2.imshow("EyeGuard Smart Eye Protection",frame)
    key = cv2.waitKey(1)

    if key == ord('q') or key == 27:   # q or ESC
        print("Exiting EyeGuard...")
        break

cap.release()
cv2.destroyAllWindows()
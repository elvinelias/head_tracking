import cv2
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python import vision
from pynput.keyboard import Controller
import math

keyboard = Controller()

MODEL_PATH = "models/face_landmarker.task"

options = vision.FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=vision.RunningMode.VIDEO
)

landmarker = vision.FaceLandmarker.create_from_options(options)

cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)

frame_index = 0

center_x = None
center_y = None

THRESHOLD = 40
SMOOTHING = 0.7

smooth_x = None
smooth_y = None

# key state tracking
w_pressed = False
a_pressed = False
s_pressed = False
d_pressed = False

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb_frame
    )

    result = landmarker.detect_for_video(mp_image, frame_index)
    frame_index += 1

    action = "CENTER"
    dx = 0
    dy = 0

    if result.face_landmarks:

        face = result.face_landmarks[0]
        nose = face[1]

        x = int(nose.x * frame.shape[1])
        y = int(nose.y * frame.shape[0])

        # smoothing
        if smooth_x is None:
            smooth_x = x
            smooth_y = y
        else:
            smooth_x = int(SMOOTHING * smooth_x + (1 - SMOOTHING) * x)
            smooth_y = int(SMOOTHING * smooth_y + (1 - SMOOTHING) * y)

        x = smooth_x
        y = smooth_y

        # set center
        if center_x is None:
            center_x = x
            center_y = y

        dx = x - center_x
        dy = y - center_y

        # draw movement line
        cv2.line(frame, (center_x, center_y), (x, y), (0,255,255), 2)

        # draw head position
        cv2.circle(frame, (x, y), 6, (0,255,0), -1)

        # draw center point
        cv2.circle(frame, (center_x, center_y), 6, (255,0,0), -1)

        # draw deadzone
        cv2.circle(frame, (center_x, center_y), THRESHOLD, (200,200,200), 1)

        # RIGHT
        if dx > THRESHOLD:
            if not d_pressed:
                keyboard.press('d')
                d_pressed = True
            action = "RIGHT"

        else:
            if d_pressed:
                keyboard.release('d')
                d_pressed = False

        # LEFT
        if dx < -THRESHOLD:
            if not a_pressed:
                keyboard.press('a')
                a_pressed = True
            action = "LEFT"

        else:
            if a_pressed:
                keyboard.release('a')
                a_pressed = False

        # DOWN
        if dy > THRESHOLD:
            if not s_pressed:
                keyboard.press('s')
                s_pressed = True
            action = "DOWN"

        else:
            if s_pressed:
                keyboard.release('s')
                s_pressed = False

        # UP
        if dy < -THRESHOLD:
            if not w_pressed:
                keyboard.press('w')
                w_pressed = True
            action = "UP"

        else:
            if w_pressed:
                keyboard.release('w')
                w_pressed = False

    # ---------- STATUS PANEL ----------
    panel_color = (40,40,40)
    cv2.rectangle(frame, (10,10), (260,150), panel_color, -1)

    cv2.putText(frame, f"Action: {action}", (20,40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

    cv2.putText(frame, f"DX: {dx}", (20,75),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)

    cv2.putText(frame, f"DY: {dy}", (20,105),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)

    cv2.putText(frame, "C = Recenter", (20,130),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1)

    cv2.putText(frame, "ESC = Quit", (150,130),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1)

    # ---------- DIRECTION ARROWS ----------
    h, w = frame.shape[:2]

    cv2.putText(frame, "W", (w//2 - 10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

    cv2.putText(frame, "A", (60, h//2),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

    cv2.putText(frame, "D", (w-80, h//2),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

    cv2.putText(frame, "S", (w//2 - 10, h-40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

    cv2.imshow("Head Motion Controller", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == 27:
        break

    if key == ord('c'):
        center_x = None
        center_y = None
        print("Recentered")

cap.release()
cv2.destroyAllWindows()
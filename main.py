import cv2
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python import vision
from pynput.keyboard import Controller

# Keyboard controller
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

while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb_frame
    )

    result = landmarker.detect_for_video(mp_image, frame_index)
    frame_index += 1

    if result.face_landmarks:

        face = result.face_landmarks[0]

        # Nose landmark (index 1)
        nose = face[1]

        x = int(nose.x * frame.shape[1])
        y = int(nose.y * frame.shape[0])

        cv2.circle(frame, (x, y), 6, (0,255,0), -1)

        if center_x is None:
            center_x = x
            center_y = y

        dx = x - center_x
        dy = y - center_y

        action = "CENTER"

        if dx > THRESHOLD:
            keyboard.press('d')
            action = "RIGHT (D)"
        elif dx < -THRESHOLD:
            keyboard.press('a')
            action = "LEFT (A)"

        if dy > THRESHOLD:
            keyboard.press('s')
            action = "DOWN (S)"
        elif dy < -THRESHOLD:
            keyboard.press('w')
            action = "UP (W)"

        cv2.putText(frame, action, (30,50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0,255,0), 2)

    cv2.imshow("Head Motion Controller", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()

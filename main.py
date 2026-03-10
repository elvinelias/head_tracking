import cv2
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python import vision

# Path to model
MODEL_PATH = "models/face_landmarker.task"

# Create FaceLandmarker options
options = vision.FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=vision.RunningMode.VIDEO
)

# Create the landmarker
landmarker = vision.FaceLandmarker.create_from_options(options)

# Start webcam (AVFOUNDATION works best on macOS)
cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)

frame_index = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture webcam frame.")
        break

    # Convert BGR → RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Convert OpenCV image to MediaPipe Image
    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb_frame
    )

    # Detect landmarks
    result = landmarker.detect_for_video(mp_image, frame_index)
    frame_index += 1

    # Draw landmarks
    if result.face_landmarks:
        for face in result.face_landmarks:
            for landmark in face:
                x = int(landmark.x * frame.shape[1])
                y = int(landmark.y * frame.shape[0])
                cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)

    cv2.imshow("Head Motion Tracker", frame)

    # ESC to quit
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
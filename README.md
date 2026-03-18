# Head Motion Game Controller

Control any game using your head movements. This project uses computer vision and real-time face tracking to convert head motion into keyboard inputs like WASD.

## Features

Real-time face tracking using MediaPipe

Head movement → keyboard controls (WASD)

Low latency and lightweight

Built with OpenCV

Keyboard control via pynput

## How It Works:

Your webcam captures your face

The program tracks your nose position

Movement is calculated relative to a center point

Directions are mapped to keys:

Movement	Key
Look Left	A
Look Right	D
Look Up	W
Look Down	S

## Installation
1. Clone the repo
git clone https://github.com/yourusername/head-motion-controller.git
cd head-motion-controller
2. Install dependencies
pip install -r requirements.txt
3. Download the model

Download the MediaPipe face model and place it here:

models/face_landmarker.task
▶️ Run the Project
python main.py
## Controls

Press ESC or Q to quit

Keep your face centered to calibrate

Move your head to control movement

🏗 Build as an App (Mac)

Using PyInstaller:

pyinstaller --windowed \
--name HeadMotionController \
--add-data "models/face_landmarker.task:models" \
--hidden-import=pynput \
--hidden-import=pynput.keyboard \
--collect-all mediapipe \
main.py

Your app will appear in:

dist/HeadMotionController.app
## Known Issues

Camera permissions may be required on macOS

Lighting conditions affect tracking accuracy

MediaPipe packaging can be tricky when building apps

High sensitivity may cause unintended movement

## Project Structure
head_motion_controller/
│
├── main.py
├── tracker.py
├── requirements.txt
├── README.md
└── models/
    └── face_landmarker.task

## License

MIT License
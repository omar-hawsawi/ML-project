import os
import subprocess
import sys

BASE = os.path.dirname(os.path.abspath(__file__))
VENV_PYTHON = os.path.join(BASE, "venv", "Scripts", "python.exe")

if (
    os.path.isfile(VENV_PYTHON)
    and os.path.normcase(os.path.abspath(sys.executable))
    != os.path.normcase(os.path.abspath(VENV_PYTHON))
):
    print(f"Starting camera with the project environment: {VENV_PYTHON}", flush=True)
    result = subprocess.call(
        [VENV_PYTHON, os.path.abspath(__file__)] + sys.argv[1:],
        cwd=BASE,
    )
    raise SystemExit(result)

from tensorflow.keras.models import load_model
import numpy as np
import cv2
import mediapipe as mp

def label_to_letter(i):
    # labels 0-8 = A-I, 9 is unused (J), labels 10-24 = K-Y
    if i < 9:
        return "ABCDEFGHI"[i]
    elif i >= 10:
        return "KLMNOPQRSTUVWXY"[i - 10]
    return "?"


def preprocess_hand(roi, hand_landmarks):
    height, width = roi.shape[:2]
    points = np.array(
        [
            (int(point.x * (width - 1)), int(point.y * (height - 1)))
            for point in hand_landmarks.landmark
        ],
        dtype=np.int32,
    )
    x_min, y_min = points.min(axis=0)
    x_max, y_max = points.max(axis=0)
    padding = max(4, int(max(x_max - x_min, y_max - y_min) * 0.15))
    x_min = max(0, x_min - padding)
    y_min = max(0, y_min - padding)
    x_max = min(width, x_max + padding + 1)
    y_max = min(height, y_max + padding + 1)

    hand_crop = roi[y_min:y_max, x_min:x_max]
    crop_height, crop_width = hand_crop.shape[:2]
    side = max(crop_height, crop_width)
    top = (side - crop_height) // 2
    bottom = side - crop_height - top
    left = (side - crop_width) // 2
    right = side - crop_width - left
    hand_crop = cv2.copyMakeBorder(
        hand_crop, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(0, 0, 0)
    )
    gray = cv2.cvtColor(hand_crop, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (28, 28), interpolation=cv2.INTER_AREA)
    img = gray.astype("float32") / 255.0
    return img.reshape(1, 28, 28, 1), (x_min, y_min, x_max, y_max)


model = load_model(os.path.join(BASE, 'sign_language_model.h5'))

train_dir = os.path.join(BASE, 'sign_mnist_train.csv')
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hand = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)

cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)

while True:
    ret, frame = cam.read()
    if not ret:
        print("Failed to grab frame")
        break

    frame = cv2.flip(frame, 1)  # Mirror the frame for display and detection
    h, w, _ = frame.shape
    size = 300
    x1, y1 = w // 2 - size // 2, h // 2 - size // 2
    x2, y2 = x1 + size, y1 + size
    roi = frame[y1:y2, x1:x2].copy()
    rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)   
    results = hand.process(rgb)

    if results.multi_hand_landmarks:
        # Draw on the ROI view so the crop-relative landmark coordinates line up.
        landmark_view = frame[y1:y2, x1:x2]
        hand_lms = results.multi_hand_landmarks[0]
        mp_drawing.draw_landmarks(
            landmark_view,
            hand_lms,
            mp_hands.HAND_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=2, circle_radius=3),
            mp_drawing.DrawingSpec(color=(255, 100, 0), thickness=2),
        )

        img, (hand_x1, hand_y1, hand_x2, hand_y2) = preprocess_hand(roi, hand_lms)
        cv2.rectangle(
            frame,
            (x1 + hand_x1, y1 + hand_y1),
            (x1 + hand_x2, y1 + hand_y2),
            (255, 0, 0),
            2,
        )
        probs = model.predict(img, verbose=0)[0]
        idx = np.argmax(probs)
        label = f"{label_to_letter(idx)} ({probs[idx] * 100:.0f}%)"
        label_color = (0, 255, 0)
    else:
        label = "Show one hand in the box"
        label_color = (0, 165, 255)

    cv2.rectangle(frame, (x1, y1), (x2, y2), label_color, 2)
    cv2.putText(frame, label, (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, label_color, 2)
   

    cv2.imshow("Camera", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):   # press q to quit
        break

cam.release()
hand.close()
cv2.destroyAllWindows()
    

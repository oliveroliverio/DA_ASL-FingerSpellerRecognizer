import cv2
import mediapipe as mp
import numpy as np
import time

from sklearn.neighbors import KNeighborsClassifier


# --------------------------------------------------
# Load training data
# --------------------------------------------------

data = np.load("training_data.npz")

X = data["X"]
y = data["y"]

print(f"Loaded {len(X)} samples.")


# --------------------------------------------------
# Train KNN classifier
# --------------------------------------------------

classifier = KNeighborsClassifier(n_neighbors=5)
classifier.fit(X, y)

print("KNN classifier trained.")


# --------------------------------------------------
# MediaPipe setup
# --------------------------------------------------

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path="hand_landmarker.task"
    ),
    running_mode=RunningMode.VIDEO,
    num_hands=1,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5,
)

landmarker = HandLandmarker.create_from_options(options)


# --------------------------------------------------
# Feature extraction
# Must match training pipeline
# --------------------------------------------------

def make_features(landmarks):
    points = np.array(
        [[lm.x, lm.y, lm.z] for lm in landmarks],
        dtype=np.float32,
    )

    # Wrist landmark becomes the origin
    points -= points[0]

    # Normalize for approximate hand size
    scale = np.max(
        np.linalg.norm(points[:, :2], axis=1)
    )

    if scale > 0:
        points /= scale

    # 21 landmarks × 3 coordinates = 63 features
    return points.flatten()


# --------------------------------------------------
# Webcam
# --------------------------------------------------

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERROR: Could not open webcam.")
    landmarker.close()
    raise SystemExit

start_time = time.time()


# --------------------------------------------------
# Main loop
# --------------------------------------------------

try:
    while True:

        success, frame = cap.read()

        if not success:
            print("Could not read webcam frame.")
            break

        # Mirror webcam
        frame = cv2.flip(frame, 1)

        # OpenCV BGR -> RGB
        rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB,
        )

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb,
        )

        timestamp_ms = int(
            (time.time() - start_time) * 1000
        )

        result = landmarker.detect_for_video(
            mp_image,
            timestamp_ms,
        )

        prediction = "NO HAND"

        if result.hand_landmarks:

            landmarks = result.hand_landmarks[0]

            features = make_features(landmarks)

            prediction = classifier.predict(
                [features]
            )[0]

            # Draw landmarks
            h, w, _ = frame.shape

            for lm in landmarks:
                x = int(lm.x * w)
                y_coord = int(lm.y * h)

                cv2.circle(
                    frame,
                    (x, y_coord),
                    5,
                    (0, 255, 0),
                    -1,
                )

        # --------------------------------------------------
        # Display prediction
        # --------------------------------------------------

        cv2.putText(
            frame,
            f"Prediction: {prediction}",
            (30, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            (0, 255, 255),
            3,
        )

        cv2.putText(
            frame,
            "Q = quit",
            (30, 105),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )

        cv2.imshow(
            "Sign Recognition",
            frame,
        )

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

finally:
    cap.release()
    cv2.destroyAllWindows()
    landmarker.close()


print("Done.")
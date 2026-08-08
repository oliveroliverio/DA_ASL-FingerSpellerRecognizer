import cv2
import mediapipe as mp
import numpy as np
import time
import urllib.request

from pathlib import Path


# --------------------------------------------------
# Paths / constants
# --------------------------------------------------

MODEL_PATH = Path("hand_landmarker.task")
DATA_PATH = Path("test_data.npz")

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/"
    "hand_landmarker.task"
)

LABELS = [
    "F",
    "O",
    "U",
    "N",
    "D",
    "A",
    "T",
    "I",
    "L",
    "SPACE",
    "BACKSPACE",
    "CLEAR",
]


# --------------------------------------------------
# Download MediaPipe model if needed
# --------------------------------------------------

if not MODEL_PATH.exists():
    print("Downloading MediaPipe hand landmark model...")

    urllib.request.urlretrieve(
        MODEL_URL,
        MODEL_PATH,
    )

    print(f"Downloaded: {MODEL_PATH}")


# --------------------------------------------------
# MediaPipe setup
# --------------------------------------------------

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path=str(MODEL_PATH)
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
# --------------------------------------------------

def make_features(landmarks):
    """
    Convert 21 MediaPipe hand landmarks into a normalized
    63-dimensional feature vector.
    """

    points = np.array(
        [
            [lm.x, lm.y, lm.z]
            for lm in landmarks
        ],
        dtype=np.float32,
    )

    # Make wrist landmark (0) the origin.
    points -= points[0]

    # Normalize based on hand size in x/y.
    scale = np.max(
        np.linalg.norm(
            points[:, :2],
            axis=1,
        )
    )

    if scale > 0:
        points /= scale

    # 21 landmarks × 3 coordinates = 63 features.
    return points.flatten()


# --------------------------------------------------
# Load existing training data if available
# --------------------------------------------------

if DATA_PATH.exists():
    data = np.load(DATA_PATH)

    X = list(data["X"])
    y = list(data["y"])

    print(
        f"Loaded {len(X)} existing samples."
    )

else:
    X = []
    y = []

    print(
        "No existing training data found."
    )


# --------------------------------------------------
# Training state
# --------------------------------------------------

current_label = "F"


# --------------------------------------------------
# Instructions
# --------------------------------------------------

print()
print("SIGN TEST DATA COLLECTION MODE")
print("==================")
print()

print("1 = F")
print("2 = O")
print("3 = U")
print("4 = N")
print("5 = D")
print("6 = A")
print("7 = T")
print("8 = I")
print("9 = L")

print()

print("S = SPACE")
print("B = BACKSPACE")
print("C = CLEAR")

print()

print("SPACEBAR = capture sample")
print("Q        = save and quit")
print()


# --------------------------------------------------
# Webcam
# --------------------------------------------------

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print(
        "ERROR: Could not open webcam."
    )

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
            print(
                "Could not read webcam frame."
            )
            break


        # --------------------------------------------------
        # Mirror webcam image
        # --------------------------------------------------

        frame = cv2.flip(
            frame,
            1,
        )


        # --------------------------------------------------
        # OpenCV BGR -> RGB
        # --------------------------------------------------

        rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB,
        )


        # --------------------------------------------------
        # Create MediaPipe image
        # --------------------------------------------------

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb,
        )


        timestamp_ms = int(
            (time.time() - start_time)
            * 1000
        )


        # --------------------------------------------------
        # Detect hand
        # --------------------------------------------------

        result = landmarker.detect_for_video(
            mp_image,
            timestamp_ms,
        )

        landmarks = None
        features = None


        # --------------------------------------------------
        # Hand detected
        # --------------------------------------------------

        if result.hand_landmarks:

            landmarks = result.hand_landmarks[0]

            features = make_features(
                landmarks
            )

            h, w, _ = frame.shape


            # --------------------------------------------------
            # Draw landmarks
            # --------------------------------------------------

            for lm in landmarks:

                x = int(
                    lm.x * w
                )

                y_coord = int(
                    lm.y * h
                )

                cv2.circle(
                    frame,
                    (x, y_coord),
                    5,
                    (0, 255, 0),
                    -1,
                )


            cv2.putText(
                frame,
                "HAND DETECTED",
                (30, 165),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )


        # --------------------------------------------------
        # No hand detected
        # --------------------------------------------------

        else:

            cv2.putText(
                frame,
                "NO HAND DETECTED",
                (30, 165),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2,
            )


        # --------------------------------------------------
        # UI text
        # --------------------------------------------------

        sample_count = y.count(
            current_label
        )


        cv2.putText(
            frame,
            f"TRAINING: {current_label}",
            (30, 45),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 255),
            2,
        )


        cv2.putText(
            frame,
            f"Samples: {sample_count}",
            (30, 85),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
        )


        cv2.putText(
            frame,
            "SPACEBAR = capture | Q = save + quit",
            (30, 125),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
        )


        cv2.imshow(
            "Sign Type Trainer",
            frame,
        )


        # --------------------------------------------------
        # Keyboard input
        # --------------------------------------------------

        key = cv2.waitKey(1) & 0xFF


        # --------------------------------------------------
        # Number keys 1-9 select alphabet labels
        # --------------------------------------------------

        if ord("1") <= key <= ord("9"):

            index = key - ord("1")

            current_label = LABELS[index]

            print(
                f"\nNow training: "
                f"{current_label}"
            )


        # --------------------------------------------------
        # S selects SPACE gesture
        # --------------------------------------------------

        elif key == ord("s"):

            current_label = "SPACE"

            print(
                f"\nNow training: "
                f"{current_label}"
            )


        # --------------------------------------------------
        # B selects BACKSPACE gesture
        # --------------------------------------------------

        elif key == ord("b"):

            current_label = "BACKSPACE"

            print(
                f"\nNow training: "
                f"{current_label}"
            )


        # --------------------------------------------------
        # C selects CLEAR gesture
        # --------------------------------------------------

        elif key == ord("c"):

            current_label = "CLEAR"

            print(
                f"\nNow training: "
                f"{current_label}"
            )


        # --------------------------------------------------
        # SPACEBAR captures current handshape
        # --------------------------------------------------

        elif key == ord(" "):

            if features is not None:

                X.append(
                    features.copy()
                )

                y.append(
                    current_label
                )

                count = y.count(
                    current_label
                )

                print(
                    f"Captured "
                    f"{current_label} "
                    f"({count} samples)"
                )

            else:

                print(
                    "No hand detected — "
                    "sample not captured."
                )


        # --------------------------------------------------
        # Q saves and quits
        # --------------------------------------------------

        elif key == ord("q"):

            print(
                "\nQ pressed. Saving..."
            )

            break


except KeyboardInterrupt:

    print(
        "\nCtrl-C detected. "
        "Saving before exit..."
    )


# --------------------------------------------------
# Cleanup
# --------------------------------------------------

cap.release()
cv2.destroyAllWindows()
landmarker.close()


# --------------------------------------------------
# Save training data
# --------------------------------------------------

if len(X) > 0:

    np.savez(
        DATA_PATH,
        X=np.array(X),
        y=np.array(y),
    )

    print(
        f"\nSaved training data to: "
        f"{DATA_PATH}"
    )

else:

    print(
        "\nNo training samples to save."
    )


# --------------------------------------------------
# Summary
# --------------------------------------------------

print()
print("TRAINING SUMMARY")
print("================")
print(
    f"Total samples: {len(X)}"
)
print()


for label in LABELS:

    print(
        f"{label}: "
        f"{y.count(label)}"
    )


print()
print("Done.")
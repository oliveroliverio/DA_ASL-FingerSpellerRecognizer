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
        np.linalg.norm(
            points[:, :2],
            axis=1,
        )
    )

    if scale > 0:
        points /= scale

    # 21 landmarks × 3 coordinates = 63 features
    return points.flatten()


# --------------------------------------------------
# Temporal stability tracking
# --------------------------------------------------

class StabilityTracker:
    def __init__(self, stability_seconds=0.75):
        self.stability_seconds = stability_seconds
        self.candidate_label = None
        self.candidate_start_time = None
        self.latched_label = None

    def update(self, prediction):
        now = time.time()

        # Prediction changed:
        # begin tracking the new candidate.
        if prediction != self.candidate_label:
            self.candidate_label = prediction
            self.candidate_start_time = now

            # Changing away from the previously accepted
            # label releases the latch.
            if prediction != self.latched_label:
                self.latched_label = None

            return None

        # Already accepted this continuously held label.
        # Do not accept it again.
        if prediction == self.latched_label:
            return None

        # Same prediction is still being held:
        # check how long it has been stable.
        if self.candidate_start_time is not None:
            held_time = now - self.candidate_start_time

            if held_time >= self.stability_seconds:
                self.latched_label = prediction
                return prediction

        return None


stability_tracker = StabilityTracker(
    stability_seconds=0.75
)


# --------------------------------------------------
# Text buffer
# --------------------------------------------------

text_buffer = ""


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

        # --------------------------------------------------
        # Mirror webcam
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
        # Hand detection
        # --------------------------------------------------

        result = landmarker.detect_for_video(
            mp_image,
            timestamp_ms,
        )

        prediction = "NO HAND"
        accepted_label = None


        # --------------------------------------------------
        # Hand detected
        # --------------------------------------------------

        if result.hand_landmarks:

            landmarks = result.hand_landmarks[0]

            features = make_features(
                landmarks
            )

            prediction = classifier.predict(
                [features]
            )[0]


            # --------------------------------------------------
            # Temporal stability logic
            # --------------------------------------------------

            accepted_label = stability_tracker.update(
                prediction
            )


            # --------------------------------------------------
            # Handle accepted label
            # --------------------------------------------------

            if accepted_label is not None:

                if accepted_label == "SPACE":
                    text_buffer += " "

                elif accepted_label == "BACKSPACE":
                    text_buffer = text_buffer[:-1]

                elif accepted_label == "CLEAR":
                    text_buffer = ""

                else:
                    text_buffer += accepted_label

                print(
                    f"ACCEPTED: {accepted_label}"
                )

                print(
                    f"TEXT: {text_buffer}"
                )


            # --------------------------------------------------
            # Draw landmarks
            # --------------------------------------------------

            h, w, _ = frame.shape

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


        # --------------------------------------------------
        # No hand detected
        # --------------------------------------------------

        else:

            # Feed NO HAND into the tracker so the previous
            # candidate/latch state is released.
            stability_tracker.update(
                "NO HAND"
            )


        # --------------------------------------------------
        # Display current prediction
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


        # --------------------------------------------------
        # Display accepted label
        # --------------------------------------------------

        if accepted_label is not None:

            cv2.putText(
                frame,
                f"Accepted: {accepted_label}",
                (30, 110),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
            )


        # --------------------------------------------------
        # Controls
        # --------------------------------------------------

        cv2.putText(
            frame,
            "Q = quit",
            (30, 155),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )


        # --------------------------------------------------
        # Display accumulated text
        # --------------------------------------------------

        cv2.putText(
            frame,
            f"Text: {text_buffer}",
            (30, 205),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (255, 255, 255),
            2,
        )


        # --------------------------------------------------
        # Show webcam window
        # --------------------------------------------------

        cv2.imshow(
            "Sign Type Live",
            frame,
        )


        # --------------------------------------------------
        # Keyboard input
        # --------------------------------------------------

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break


finally:

    cap.release()
    cv2.destroyAllWindows()
    landmarker.close()


print("Done.")
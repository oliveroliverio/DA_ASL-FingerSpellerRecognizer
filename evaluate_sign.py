import cv2
import mediapipe as mp
import numpy as np
import time

from collections import Counter, defaultdict
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.neighbors import KNeighborsClassifier


# --------------------------------------------------
# Configuration
# --------------------------------------------------

TEST_LABELS = [
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

TRIALS_PER_LABEL = 5
STABILITY_SECONDS = 0.75


# --------------------------------------------------
# Load training data
# --------------------------------------------------

data = np.load("training_data.npz")

X = data["X"]
y = data["y"]

print(f"Loaded {len(X)} training samples.")


# --------------------------------------------------
# Train KNN baseline
# --------------------------------------------------

classifier = KNeighborsClassifier(
    n_neighbors=5
)

classifier.fit(
    X,
    y,
)

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

landmarker = HandLandmarker.create_from_options(
    options
)


# --------------------------------------------------
# Feature extraction
# Must match training pipeline
# --------------------------------------------------

def make_features(landmarks):
    points = np.array(
        [
            [lm.x, lm.y, lm.z]
            for lm in landmarks
        ],
        dtype=np.float32,
    )

    # Wrist becomes origin
    points -= points[0]

    # Normalize by approximate hand size
    scale = np.max(
        np.linalg.norm(
            points[:, :2],
            axis=1,
        )
    )

    if scale > 0:
        points /= scale

    return points.flatten()


# --------------------------------------------------
# Stability tracker
# --------------------------------------------------

class StabilityTracker:
    def __init__(
        self,
        stability_seconds=0.75,
    ):
        self.stability_seconds = stability_seconds
        self.candidate_label = None
        self.candidate_start_time = None
        self.latched_label = None

    def reset(self):
        self.candidate_label = None
        self.candidate_start_time = None
        self.latched_label = None

    def update(self, prediction):
        now = time.time()

        # New candidate
        if prediction != self.candidate_label:
            self.candidate_label = prediction
            self.candidate_start_time = now

            if prediction != self.latched_label:
                self.latched_label = None

            return None

        # Already accepted this held prediction
        if prediction == self.latched_label:
            return None

        if self.candidate_start_time is not None:
            held_time = (
                now
                - self.candidate_start_time
            )

            if held_time >= self.stability_seconds:
                self.latched_label = prediction
                return prediction

        return None


stability_tracker = StabilityTracker(
    stability_seconds=STABILITY_SECONDS
)


# --------------------------------------------------
# Evaluation state
# --------------------------------------------------

current_label_index = 0
current_trial = 0

y_true = []
y_pred = []

per_label_results = defaultdict(list)

waiting_for_reset = False


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
# Helper functions
# --------------------------------------------------

def current_target():
    return TEST_LABELS[
        current_label_index
    ]


def print_trial_result(
    target,
    prediction,
    trial_number,
):
    correct = (
        target == prediction
    )

    symbol = "✓" if correct else "✗"

    print(
        f"{target} "
        f"trial {trial_number}: "
        f"predicted {prediction} "
        f"{symbol}"
    )


def print_final_results():
    print()
    print("=" * 60)
    print("EVALUATION COMPLETE")
    print("=" * 60)
    print()

    total = len(y_true)

    correct = sum(
        actual == predicted
        for actual, predicted
        in zip(y_true, y_pred)
    )

    accuracy = (
        correct / total
        if total > 0
        else 0
    )

    print(
        f"Overall accuracy: "
        f"{correct}/{total} "
        f"= {accuracy:.1%}"
    )

    print()
    print("PER-LABEL RESULTS")
    print("-----------------")

    for label in TEST_LABELS:
        results = (
            per_label_results[label]
        )

        label_correct = sum(
            result == label
            for result in results
        )

        label_total = len(results)

        label_accuracy = (
            label_correct / label_total
            if label_total > 0
            else 0
        )

        print(
            f"{label:10s} "
            f"{label_correct}/{label_total} "
            f"= {label_accuracy:.1%}"
        )

    print()
    print("PREDICTION COUNTS")
    print("-----------------")

    print(
        Counter(y_pred)
    )

    print()
    print("CONFUSION MATRIX")
    print("----------------")

    matrix = confusion_matrix(
        y_true,
        y_pred,
        labels=TEST_LABELS,
    )

    header = (
        "true\\pred".ljust(12)
        + " ".join(
            label[:5].rjust(6)
            for label in TEST_LABELS
        )
    )

    print(header)

    for label, row in zip(
        TEST_LABELS,
        matrix,
    ):
        row_text = " ".join(
            str(value).rjust(6)
            for value in row
        )

        print(
            label[:10].ljust(12)
            + row_text
        )

    print()
    print("CLASSIFICATION REPORT")
    print("---------------------")

    print(
        classification_report(
            y_true,
            y_pred,
            labels=TEST_LABELS,
            zero_division=0,
        )
    )


# --------------------------------------------------
# Instructions
# --------------------------------------------------

print()
print("SIGN EVALUATION MODE")
print("====================")
print()
print(
    f"{TRIALS_PER_LABEL} trials "
    f"per label"
)
print()

print(
    "Hold the requested gesture "
    "until it is accepted."
)

print(
    "After each trial, remove your "
    "hand briefly before the next."
)

print()

print(
    f"Starting with: "
    f"{current_target()}"
)

print()


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

        frame = cv2.flip(
            frame,
            1,
        )

        rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB,
        )

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb,
        )

        timestamp_ms = int(
            (time.time() - start_time)
            * 1000
        )

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

            landmarks = (
                result.hand_landmarks[0]
            )

            features = make_features(
                landmarks
            )

            prediction = (
                classifier.predict(
                    [features]
                )[0]
            )

            # Only allow a new trial after
            # the hand has been removed.
            if not waiting_for_reset:
                accepted_label = (
                    stability_tracker.update(
                        prediction
                    )
                )

            # Draw landmarks
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

            stability_tracker.update(
                "NO HAND"
            )

            # Hand removal arms the next trial.
            if waiting_for_reset:
                waiting_for_reset = False
                stability_tracker.reset()

        # --------------------------------------------------
        # Accepted evaluation trial
        # --------------------------------------------------

        if accepted_label is not None:

            target = current_target()

            current_trial += 1

            y_true.append(
                target
            )

            y_pred.append(
                accepted_label
            )

            per_label_results[
                target
            ].append(
                accepted_label
            )

            print_trial_result(
                target=target,
                prediction=accepted_label,
                trial_number=current_trial,
            )

            waiting_for_reset = True

            # Finished all trials for this label
            if (
                current_trial
                >= TRIALS_PER_LABEL
            ):
                current_label_index += 1
                current_trial = 0

                # Entire evaluation finished
                if (
                    current_label_index
                    >= len(TEST_LABELS)
                ):
                    break

                print()
                print(
                    f"NEXT LABEL: "
                    f"{current_target()}"
                )
                print()

        # --------------------------------------------------
        # UI
        # --------------------------------------------------

        target_text = (
            current_target()
            if current_label_index
            < len(TEST_LABELS)
            else "DONE"
        )

        cv2.putText(
            frame,
            f"TARGET: {target_text}",
            (30, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.1,
            (0, 255, 255),
            3,
        )

        cv2.putText(
            frame,
            (
                f"Prediction: "
                f"{prediction}"
            ),
            (30, 95),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
        )

        if (
            current_label_index
            < len(TEST_LABELS)
        ):
            cv2.putText(
                frame,
                (
                    f"Trial: "
                    f"{current_trial + 1}"
                    f"/{TRIALS_PER_LABEL}"
                ),
                (30, 135),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
            )

        if waiting_for_reset:
            cv2.putText(
                frame,
                "REMOVE HAND",
                (30, 180),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 0, 255),
                2,
            )

        else:
            cv2.putText(
                frame,
                "HOLD TARGET GESTURE",
                (30, 180),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
            )

        cv2.putText(
            frame,
            "Q = quit",
            (30, 225),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )

        cv2.imshow(
            "Sign Evaluation",
            frame,
        )

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            print(
                "\nEvaluation cancelled."
            )
            break


finally:
    cap.release()
    cv2.destroyAllWindows()
    landmarker.close()


# --------------------------------------------------
# Final report
# --------------------------------------------------

if len(y_true) > 0:
    print_final_results()

print("Done.")
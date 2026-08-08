from datetime import date
from pathlib import Path

import numpy as np

from sklearn.metrics import classification_report, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier


# --------------------------------------------------
# Paths
# --------------------------------------------------

TRAIN_PATH = "training_data.npz"
TEST_PATH = "test_data.npz"

RESULTS_DIR = Path("results")

REPORT_PATH = (
    RESULTS_DIR
    / f"evaluation_knn_heldout_{date.today().isoformat()}.md"
)


# --------------------------------------------------
# Configuration
# --------------------------------------------------

K_NEIGHBORS = 5


# --------------------------------------------------
# Load datasets
# --------------------------------------------------

train_data = np.load(TRAIN_PATH)
test_data = np.load(TEST_PATH)

X_train = train_data["X"]
y_train = train_data["y"]

X_test = test_data["X"]
y_test = test_data["y"]

print(f"Training samples: {len(X_train)}")
print(f"Test samples: {len(X_test)}")


# --------------------------------------------------
# Train KNN
# --------------------------------------------------

classifier = KNeighborsClassifier(
    n_neighbors=K_NEIGHBORS
)

classifier.fit(
    X_train,
    y_train,
)

print("KNN classifier trained.")


# --------------------------------------------------
# Predict held-out test set
# --------------------------------------------------

y_pred = classifier.predict(
    X_test
)


# --------------------------------------------------
# Accuracy
# --------------------------------------------------

correct = int(
    np.sum(
        y_pred == y_test
    )
)

total = len(y_test)

accuracy = (
    correct / total
    if total > 0
    else 0
)


# --------------------------------------------------
# Labels
# --------------------------------------------------

labels = sorted(
    np.unique(
        np.concatenate(
            [
                y_train,
                y_test,
            ]
        )
    )
)


# --------------------------------------------------
# Classification report
# --------------------------------------------------

classification_report_text = classification_report(
    y_test,
    y_pred,
    labels=labels,
    zero_division=0,
)


# --------------------------------------------------
# Confusion matrix
# --------------------------------------------------

matrix = confusion_matrix(
    y_test,
    y_pred,
    labels=labels,
)

header = (
    "true\\pred".ljust(12)
    + " ".join(
        label[:5].rjust(6)
        for label in labels
    )
)

confusion_matrix_lines = [
    header
]

for label, row in zip(
    labels,
    matrix,
):
    row_text = " ".join(
        str(value).rjust(6)
        for value in row
    )

    confusion_matrix_lines.append(
        label[:10].ljust(12)
        + row_text
    )

confusion_matrix_text = "\n".join(
    confusion_matrix_lines
)


# --------------------------------------------------
# Misclassifications
# --------------------------------------------------

misclassification_lines = []

for index, (
    actual,
    predicted,
) in enumerate(
    zip(
        y_test,
        y_pred,
    )
):
    if actual != predicted:
        misclassification_lines.append(
            (
                f"Sample {index}: "
                f"actual={actual}, "
                f"predicted={predicted}"
            )
        )


if len(misclassification_lines) == 0:
    misclassification_text = (
        "No misclassifications."
    )

else:
    misclassification_text = "\n".join(
        misclassification_lines
    )


# --------------------------------------------------
# Test support by label
# --------------------------------------------------

support_lines = []

for label in labels:

    support = int(
        np.sum(
            y_test == label
        )
    )

    support_lines.append(
        f"- `{label}`: {support}"
    )

support_text = "\n".join(
    support_lines
)


# --------------------------------------------------
# Terminal report
# --------------------------------------------------

print()
print("=" * 60)
print("HELD-OUT KNN EVALUATION")
print("=" * 60)
print()

print(
    f"Overall accuracy: "
    f"{correct}/{total} "
    f"= {accuracy:.1%}"
)

print()
print("CLASSIFICATION REPORT")
print("---------------------")

print(
    classification_report_text
)

print()
print("CONFUSION MATRIX")
print("----------------")

print(
    confusion_matrix_text
)

print()
print("MISCLASSIFICATIONS")
print("------------------")

print(
    misclassification_text
)


# --------------------------------------------------
# Markdown report
# --------------------------------------------------

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

report_markdown = "\n".join(
    [
        f"# Held-Out KNN Evaluation — {date.today().isoformat()}",
        "",
        "## Setup",
        "",
        "- Model: K-Nearest Neighbors (KNN)",
        f"- `k`: {K_NEIGHBORS}",
        f"- Training samples: {len(X_train)}",
        f"- Held-out test samples: {len(X_test)}",
        f"- Training dataset: `{TRAIN_PATH}`",
        f"- Test dataset: `{TEST_PATH}`",
        (
            "- Feature representation: "
            "21 MediaPipe landmarks × 3 coordinates = 63 features"
        ),
        "- Evaluation type: offline held-out same-signer evaluation",
        "",
        "## Important Limitation",
        "",
        (
            "The test samples are stored separately from the training samples, "
            "so they were not used to fit the KNN classifier."
        ),
        "",
        (
            "However, this is still a same-signer evaluation. "
            "The training and test data were collected from the same signer "
            "and likely under similar environmental conditions."
        ),
        "",
        (
            "Therefore, this result should not be interpreted as evidence "
            "of signer-independent generalization."
        ),
        "",
        "## Overall Accuracy",
        "",
        f"**{correct}/{total} = {accuracy:.1%}**",
        "",
        "## Test Support by Label",
        "",
        support_text,
        "",
        "## Classification Report",
        "",
        "```text",
        classification_report_text.rstrip(),
        "```",
        "",
        "## Confusion Matrix",
        "",
        "```text",
        confusion_matrix_text,
        "```",
        "",
        "## Misclassifications",
        "",
        "```text",
        misclassification_text,
        "```",
        "",
        "## Interpretation",
        "",
        (
            "This evaluation measures how well the KNN classifier predicts "
            "fresh samples stored separately from the training dataset."
        ),
        "",
        (
            "Future comparisons should evaluate alternative models, including "
            "the planned PyTorch baseline, against the same held-out test set "
            "whenever appropriate."
        ),
        "",
    ]
)

REPORT_PATH.write_text(
    report_markdown,
    encoding="utf-8",
)


# --------------------------------------------------
# Final output
# --------------------------------------------------

print()
print(
    f"Markdown report saved to: "
    f"{REPORT_PATH}"
)

print()
print("Done.")
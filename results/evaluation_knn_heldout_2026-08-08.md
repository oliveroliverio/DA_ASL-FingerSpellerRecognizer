# Held-Out KNN Evaluation — 2026-08-08

## Setup

- Model: K-Nearest Neighbors (KNN)
- `k`: 5
- Training samples: 199
- Held-out test samples: 123
- Training dataset: `training_data.npz`
- Test dataset: `test_data.npz`
- Feature representation: 21 MediaPipe landmarks × 3 coordinates = 63 features
- Evaluation type: offline held-out same-signer evaluation

## Important Limitation

The test samples are stored separately from the training samples, so they were not used to fit the KNN classifier.

However, this is still a same-signer evaluation. The training and test data were collected from the same signer and likely under similar environmental conditions.

Therefore, this result should not be interpreted as evidence of signer-independent generalization.

## Overall Accuracy

**123/123 = 100.0%**

## Test Support by Label

- `A`: 10
- `BACKSPACE`: 10
- `CLEAR`: 11
- `D`: 10
- `F`: 10
- `I`: 11
- `L`: 10
- `N`: 11
- `O`: 10
- `SPACE`: 10
- `T`: 10
- `U`: 10

## Classification Report

```text
              precision    recall  f1-score   support

           A       1.00      1.00      1.00        10
   BACKSPACE       1.00      1.00      1.00        10
       CLEAR       1.00      1.00      1.00        11
           D       1.00      1.00      1.00        10
           F       1.00      1.00      1.00        10
           I       1.00      1.00      1.00        11
           L       1.00      1.00      1.00        10
           N       1.00      1.00      1.00        11
           O       1.00      1.00      1.00        10
       SPACE       1.00      1.00      1.00        10
           T       1.00      1.00      1.00        10
           U       1.00      1.00      1.00        10

    accuracy                           1.00       123
   macro avg       1.00      1.00      1.00       123
weighted avg       1.00      1.00      1.00       123
```

## Confusion Matrix

```text
true\pred        A  BACKS  CLEAR      D      F      I      L      N      O  SPACE      T      U
A               10      0      0      0      0      0      0      0      0      0      0      0
BACKSPACE        0     10      0      0      0      0      0      0      0      0      0      0
CLEAR            0      0     11      0      0      0      0      0      0      0      0      0
D                0      0      0     10      0      0      0      0      0      0      0      0
F                0      0      0      0     10      0      0      0      0      0      0      0
I                0      0      0      0      0     11      0      0      0      0      0      0
L                0      0      0      0      0      0     10      0      0      0      0      0
N                0      0      0      0      0      0      0     11      0      0      0      0
O                0      0      0      0      0      0      0      0     10      0      0      0
SPACE            0      0      0      0      0      0      0      0      0     10      0      0
T                0      0      0      0      0      0      0      0      0      0     10      0
U                0      0      0      0      0      0      0      0      0      0      0     10
```

## Misclassifications

```text
No misclassifications.
```

## Interpretation

This evaluation measures how well the KNN classifier predicts fresh samples stored separately from the training dataset.

Future comparisons should evaluate alternative models, including the planned PyTorch baseline, against the same held-out test set whenever appropriate.

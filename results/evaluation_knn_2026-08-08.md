# KNN Evaluation — 2026-08-08

## Setup
- Model: KNN
- k: 5
- Training samples: 199
- Evaluation trials: 60
- Trials per class: 5
- Classes: F, O, U, N, D, A, T, I, L, SPACE, BACKSPACE, CLEAR
- Signer: same signer used for training
- Environment: live webcam evaluation
- Stability threshold: 0.75 s

## Important Limitation
This is an interactive same-signer evaluation, not a held-out test set. The 100% accuracy should therefore be treated as a functional prototype result, not evidence of signer-independent generalization.

## Results


SIGN EVALUATION MODE
====================

5 trials per label

Hold the requested gesture until it is accepted.
After each trial, remove your hand briefly before the next.

Starting with: F

W0000 00:00:1786227751.966870 2544462 landmark_projection_calculator.cc:81] Using NORM_RECT without IMAGE_DIMENSIONS is only supported for the square ROI. Provide IMAGE_DIMENSIONS or use PROJECTION_MATRIX.
F trial 1: predicted F ✓
F trial 2: predicted F ✓
F trial 3: predicted F ✓
F trial 4: predicted F ✓
F trial 5: predicted F ✓

NEXT LABEL: O

O trial 1: predicted O ✓
O trial 2: predicted O ✓
O trial 3: predicted O ✓
O trial 4: predicted O ✓
O trial 5: predicted O ✓

NEXT LABEL: U

U trial 1: predicted U ✓
U trial 2: predicted U ✓
U trial 3: predicted U ✓
U trial 4: predicted U ✓
U trial 5: predicted U ✓

NEXT LABEL: N

N trial 1: predicted N ✓
N trial 2: predicted N ✓
N trial 3: predicted N ✓
N trial 4: predicted N ✓
N trial 5: predicted N ✓

NEXT LABEL: D

D trial 1: predicted D ✓
D trial 2: predicted D ✓
D trial 3: predicted D ✓
D trial 4: predicted D ✓
D trial 5: predicted D ✓

NEXT LABEL: A

A trial 1: predicted A ✓
A trial 2: predicted A ✓
A trial 3: predicted A ✓
A trial 4: predicted A ✓
A trial 5: predicted A ✓

NEXT LABEL: T

T trial 1: predicted T ✓
T trial 2: predicted T ✓
T trial 3: predicted T ✓
T trial 4: predicted T ✓
T trial 5: predicted T ✓

NEXT LABEL: I

I trial 1: predicted I ✓
I trial 2: predicted I ✓
I trial 3: predicted I ✓
I trial 4: predicted I ✓
I trial 5: predicted I ✓

NEXT LABEL: L

L trial 1: predicted L ✓
L trial 2: predicted L ✓
L trial 3: predicted L ✓
L trial 4: predicted L ✓
L trial 5: predicted L ✓

NEXT LABEL: SPACE

SPACE trial 1: predicted SPACE ✓
SPACE trial 2: predicted SPACE ✓
SPACE trial 3: predicted SPACE ✓
SPACE trial 4: predicted SPACE ✓
SPACE trial 5: predicted SPACE ✓

NEXT LABEL: BACKSPACE

BACKSPACE trial 1: predicted BACKSPACE ✓
BACKSPACE trial 2: predicted BACKSPACE ✓
BACKSPACE trial 3: predicted BACKSPACE ✓
BACKSPACE trial 4: predicted BACKSPACE ✓
BACKSPACE trial 5: predicted BACKSPACE ✓

NEXT LABEL: CLEAR

CLEAR trial 1: predicted CLEAR ✓
CLEAR trial 2: predicted CLEAR ✓
CLEAR trial 3: predicted CLEAR ✓
CLEAR trial 4: predicted CLEAR ✓
CLEAR trial 5: predicted CLEAR ✓

============================================================
EVALUATION COMPLETE
============================================================

Overall accuracy: 60/60 = 100.0%

PER-LABEL RESULTS
-----------------
F          5/5 = 100.0%
O          5/5 = 100.0%
U          5/5 = 100.0%
N          5/5 = 100.0%
D          5/5 = 100.0%
A          5/5 = 100.0%
T          5/5 = 100.0%
I          5/5 = 100.0%
L          5/5 = 100.0%
SPACE      5/5 = 100.0%
BACKSPACE  5/5 = 100.0%
CLEAR      5/5 = 100.0%

PREDICTION COUNTS
-----------------
Counter({np.str_('F'): 5, np.str_('O'): 5, np.str_('U'): 5, np.str_('N'): 5, np.str_('D'): 5, np.str_('A'): 5, np.str_('T'): 5, np.str_('I'): 5, np.str_('L'): 5, np.str_('SPACE'): 5, np.str_('BACKSPACE'): 5, np.str_('CLEAR'): 5})

CONFUSION MATRIX
----------------
true\pred        F      O      U      N      D      A      T      I      L  SPACE  BACKS  CLEAR
F                5      0      0      0      0      0      0      0      0      0      0      0
O                0      5      0      0      0      0      0      0      0      0      0      0
U                0      0      5      0      0      0      0      0      0      0      0      0
N                0      0      0      5      0      0      0      0      0      0      0      0
D                0      0      0      0      5      0      0      0      0      0      0      0
A                0      0      0      0      0      5      0      0      0      0      0      0
T                0      0      0      0      0      0      5      0      0      0      0      0
I                0      0      0      0      0      0      0      5      0      0      0      0
L                0      0      0      0      0      0      0      0      5      0      0      0
SPACE            0      0      0      0      0      0      0      0      0      5      0      0
BACKSPACE        0      0      0      0      0      0      0      0      0      0      5      0
CLEAR            0      0      0      0      0      0      0      0      0      0      0      5

CLASSIFICATION REPORT
---------------------
              precision    recall  f1-score   support

           F       1.00      1.00      1.00         5
           O       1.00      1.00      1.00         5
           U       1.00      1.00      1.00         5
           N       1.00      1.00      1.00         5
           D       1.00      1.00      1.00         5
           A       1.00      1.00      1.00         5
           T       1.00      1.00      1.00         5
           I       1.00      1.00      1.00         5
           L       1.00      1.00      1.00         5
       SPACE       1.00      1.00      1.00         5
   BACKSPACE       1.00      1.00      1.00         5
       CLEAR       1.00      1.00      1.00         5

    accuracy                           1.00        60
   macro avg       1.00      1.00      1.00        60
weighted avg       1.00      1.00      1.00        60

Git Commit Message

feat: add held-out evaluation pipeline for KNN baseline

- add separate test dataset collection workflow
- keep evaluation samples isolated from training data
- add offline evaluation against held-out samples
- report accuracy, per-class metrics, and confusion matrix
- save dated evaluation results for future KNN vs PyTorch comparison
# Model Performance Report

## Task
Binary classification to predict whether a **closed incident** will breach SLA.

## Evaluation split
- Training rows: **2,374**
- Test rows: **594**
- Positive rate (train): **33.4%**
- Positive rate (test): **33.3%**

## Metrics
- Accuracy: **74.58%**
- Precision: **60.63%**
- Recall: **67.68%**
- F1 Score: **63.96%**
- ROC-AUC: **82.42%**

## Confusion matrix
- True negatives: **309**
- False positives: **87**
- False negatives: **64**
- True positives: **134**

See: `figures/model_confusion_matrix.png`

## Top feature drivers
The saved file `models/top_features.csv` contains the full ranked feature list. In this synthetic setup, the strongest signals typically include:
- assignment delay hours
- affected users
- created hour
- alert count
- backlog at creation
- repeat incident flag

## Notes
This model is trained on synthetic data and is intended for portfolio demonstration rather than production deployment.

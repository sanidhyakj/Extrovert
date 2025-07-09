readme_text = """
Personality Prediction from Social Behavior Data
===============================================

This project builds a binary classification model using XGBoost to predict whether a person is an Introvert or Extrovert based on their social behavior features.

Directory Structure
-------------------
.
├── Extrovert/
│   ├── train.csv
│   ├── test.csv
│   ├── sample_submission.csv
├── cleaned_dataset.csv
├── models/
│   ├── best_model.json
│   ├── final_model.joblib
│   ├── scaler_center.npy
│   ├── scaler_scale.npy
│   ├── features.csv
├── train_model.py
├── test.py
├── submission.csv
├── comparison.csv (optional)
├── README.txt

Files Description
-----------------
1. train_model.py
   - Loads and preprocesses training data.
   - Splits into training and validation sets.
   - Scales features using RobustScaler.
   - Tunes XGBoost model using 7-fold cross-validation.
   - Saves best model and scaler.

2. test.py
   - Loads and cleans test.csv.
   - Applies saved scaler and model.
   - Predicts labels and saves to submission.csv.
   - Optionally compares predictions with sample_submission.csv if available.

Dataset
-------
Features:
- Time_spent_Alone
- Stage_fear
- Drained_after_socializing
- Social_event_attendance
- Going_outside
- Friends_circle_size
- Post_frequency

Engineered Features:
- social_index
- introvert_index
- social_ratio

Target:
- Personality (Introvert or Extrovert)

Output
------
submission.csv (example):
id,Personality
101,Introvert
102,Extrovert

comparison.csv (optional if ground truth exists):
id,Expected,Predicted

Requirements
------------
pip install pandas numpy scikit-learn xgboost matplotlib seaborn joblib

Running the Pipeline
--------------------
Train the model:
    python train_model.py

Predict on test data:
    python test.py

Evaluation
----------
- Accuracy
- ROC-AUC Score
- Classification Report
"""

with open("/mnt/data/README.txt", "w") as f:
    f.write(readme_text)

"/mnt/data/README.txt"


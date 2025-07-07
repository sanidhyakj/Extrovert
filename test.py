import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.preprocessing import StandardScaler

# Load test data
test = pd.read_csv('test.csv')  # Adjust path if needed

# Define features explicitly (same as train)
features = [
    'Time_spent_Alone',
    'Stage_fear',
    'Social_event_attendance',
    'Going_outside',
    'Drained_after_socializing',
    'Friends_circle_size',
    'Post_frequency'
]

# Load features and make copy
X_test = test[features].copy()

# Map Yes/No to 1/0 safely
for col in features:
    if X_test[col].dtype == 'object':
        X_test.loc[:, col] = X_test[col].map({'Yes': 1, 'No': 0})

# Scaling
mean_ = np.load('models/scaler_mean.npy')
scale_ = np.load('models/scaler_scale.npy')
scaler = StandardScaler()
scaler.mean_ = mean_
scaler.scale_ = scale_

X_test_scaled = scaler.transform(X_test)

# Predict
model = lgb.Booster(model_file='models/lgb_model.txt')
y_pred = model.predict(X_test_scaled)
y_pred_labels = (y_pred > 0.5).astype(int)

# Load label classes and map correctly
le_classes = pd.read_csv('models/label_classes.csv').squeeze()
label_list = le_classes.tolist()
y_pred_names = [label_list[i] for i in y_pred_labels]

# Prepare submission
submission = pd.DataFrame({
    'id': test['id'],
    'Personality': y_pred_names
})
submission.to_csv('submission.csv', index=False)
print("✅ submission.csv created.")

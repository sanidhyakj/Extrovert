import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, accuracy_score
import lightgbm as lgb
import os

# Load training data
train = pd.read_csv('train.csv')

# Encode target variable
le = LabelEncoder()
train['target'] = le.fit_transform(train['Personality'])  # Introvert=0, Extrovert=1

# Features list (exclude id and target columns)
features = [
    'Time_spent_Alone',
    'Stage_fear',
    'Social_event_attendance',
    'Going_outside',
    'Drained_after_socializing',
    'Friends_circle_size',
    'Post_frequency'
]

# Convert Yes/No to 1/0 for categorical features
for col in features:
    if train[col].dtype == 'object':
        train[col] = train[col].map({'Yes': 1, 'No': 0})

X = train[features]
y = train['target']

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split into train and validation sets
X_train, X_val, y_train, y_val = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# Set up LightGBM model and hyperparameter grid for tuning
model = lgb.LGBMClassifier(objective='binary', random_state=42, verbose=-1)

param_dist = {
    'num_leaves': [15, 31, 50],
    'learning_rate': [0.1, 0.05, 0.01],
    'n_estimators': [100, 200, 500],
    'max_depth': [5, 10, 20, -1],
    'subsample': [0.8, 1],
    'colsample_bytree': [0.8, 1],
    'min_child_samples': [20, 30, 40],  # increased to reduce warnings
}

# Use StratifiedKFold for consistent folds
cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

random_search = RandomizedSearchCV(
    estimator=model,
    param_distributions=param_dist,
    n_iter=20,  # Number of parameter settings sampled
    scoring='accuracy',
    cv=cv,
    verbose=1,
    n_jobs=-1,
    random_state=42
)

# Run randomized search to find good hyperparameters quickly
random_search.fit(X_train, y_train)

print(f"Best hyperparameters: {random_search.best_params_}")
print(f"Best CV accuracy: {random_search.best_score_:.4f}")

# Retrain with best parameters + early stopping on validation set
best_params = random_search.best_params_
final_model = lgb.LGBMClassifier(
    **best_params,
    objective='binary',
    random_state=42,
    n_estimators=1000,
    verbose=-1
)

final_model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    eval_metric='binary_logloss',
    early_stopping_rounds=50,
    verbose=-1  # suppress training logs and warnings
)

# Predict on validation set and evaluate
y_val_pred = final_model.predict(X_val)
print("Validation Classification Report:")
print(classification_report(y_val, y_val_pred, target_names=le.classes_))

acc = accuracy_score(y_val, y_val_pred)
print(f"Validation Accuracy: {acc:.4f}")

# Save model and scaler info
os.makedirs('models', exist_ok=True)
final_model.booster_.save_model('models/lgb_model.txt')
np.save('models/scaler_mean.npy', scaler.mean_)
np.save('models/scaler_scale.npy', scaler.scale_)
pd.Series(le.classes_).to_csv('models/label_classes.csv', index=False)

print("✅ Model and scaler saved.")

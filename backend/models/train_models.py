"""
================================================================================
COMPLETE MODEL TRAINING SCRIPT
================================================================================

for each prediction task (Equipment Failure & Flight Cancellation)

THE 3 MODELS:
-------------
1. Random Forest - Uses 100 decision trees voting together
2. XGBoost - Gradient boosting (builds trees sequentially, learning from mistakes)
3. LightGBM - Light gradient boosting (faster version of XGBoost)

HOW IT WORKS:
-------------
Step 1: Split Data
    - 15,000 records split into:
    - Train (70%): Learn patterns
    - Validation (15%): Compare models
    - Test (15%): Final evaluation

Step 2: Train All 3 Models
    - For each model:
        * Fit on training data
        * Predict on validation data
        * Calculate ROC-AUC score

Step 3: Pick the Winner
    - Compare ROC-AUC scores
    - Example:
        Random Forest: 0.56
        XGBoost: 0.58  ← WINNER!
        LightGBM: 0.55
    - Save only the best model

Step 4: Repeat for Both Tasks
    - Equipment Failure (Task 1)
    - Flight Cancellation (Task 2)

WHAT GETS SAVED:
----------------
Only the 2 BEST models:
- equipment_failure_model.pkl (best of 3)
- flight_cancellation_model.pkl (best of 3)

This is the PROPER ML workflow - train multiple models, compare, pick best!

================================================================================
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import confusion_matrix
import joblib
import os

print("\n" + "="*70)
print("AI MODEL TRAINING PIPELINE - FULL VERSION")
print("Training: Random Forest, XGBoost, LightGBM")
print("="*70)

# STEP 1: LOAD PREPROCESSED DATA
print("\nStep 1: Loading preprocessed data...")
data_path = 'backend/data/processed/aircraft_flight_final.csv'

if not os.path.exists(data_path):
    print(f"\nERROR: Preprocessed data not found at {data_path}")
    print("Please run: python backend/data/preprocessing.py first!")
    exit(1)

df = pd.read_csv(data_path)
print(f"   Loaded {len(df):,} records")

X = df.drop(['equipment_failure', 'flight_cancelled'], axis=1)
y_equipment = df['equipment_failure']
y_cancellation = df['flight_cancelled']

print(f"   Features: {X.shape[1]}")
print(f"   Equipment failure rate: {y_equipment.mean()*100:.1f}%")
print(f"   Cancellation rate: {y_cancellation.mean()*100:.1f}%")

# STEP 2: SPLIT DATA
print("\nStep 2: Splitting data into Train/Val/Test (70/15/15)...")

X_temp, X_test, y_temp_equip, y_test_equip = train_test_split(
    X, y_equipment, test_size=0.15, random_state=42, stratify=y_equipment
)

X_train, X_val, y_train_equip, y_val_equip = train_test_split(
    X_temp, y_temp_equip, test_size=0.176, random_state=42, stratify=y_temp_equip
)

print(f"   Train: {len(X_train):,} samples")
print(f"   Validation: {len(X_val):,} samples")
print(f"   Test: {len(X_test):,} samples")

# STEP 3: TRAIN EQUIPMENT FAILURE MODELS
print("\n" + "="*70)
print("PART 1: EQUIPMENT FAILURE PREDICTION")
print("="*70)

equipment_models = {}
equipment_scores = {}

# MODEL 1: Random Forest
print("\n[1/3] Training Random Forest...")
rf = RandomForestClassifier(
    n_estimators=100,
    max_depth=15,
    min_samples_split=5,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)
rf.fit(X_train, y_train_equip)
y_val_prob_rf = rf.predict_proba(X_val)[:, 1]
roc_auc_rf = roc_auc_score(y_val_equip, y_val_prob_rf)
equipment_models['Random Forest'] = rf
equipment_scores['Random Forest'] = roc_auc_rf
print(f"   Validation ROC-AUC: {roc_auc_rf:.4f}")

# MODEL 2: XGBoost
print("\n[2/3] Training XGBoost...")
neg_count = (y_train_equip == 0).sum()
pos_count = (y_train_equip == 1).sum()
scale_pos_weight = neg_count / pos_count

xgb = XGBClassifier(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    scale_pos_weight=scale_pos_weight,
    random_state=42,
    n_jobs=-1,
    eval_metric='logloss'
)
xgb.fit(X_train, y_train_equip)
y_val_prob_xgb = xgb.predict_proba(X_val)[:, 1]
roc_auc_xgb = roc_auc_score(y_val_equip, y_val_prob_xgb)
equipment_models['XGBoost'] = xgb
equipment_scores['XGBoost'] = roc_auc_xgb
print(f"   Validation ROC-AUC: {roc_auc_xgb:.4f}")

# MODEL 3: LightGBM
print("\n[3/3] Training LightGBM...")
lgbm = LGBMClassifier(
    n_estimators=100,
    max_depth=15,
    learning_rate=0.1,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1,
    verbose=-1
)
lgbm.fit(X_train, y_train_equip)
y_val_prob_lgbm = lgbm.predict_proba(X_val)[:, 1]
roc_auc_lgbm = roc_auc_score(y_val_equip, y_val_prob_lgbm)
equipment_models['LightGBM'] = lgbm
equipment_scores['LightGBM'] = roc_auc_lgbm
print(f"   Validation ROC-AUC: {roc_auc_lgbm:.4f}")

# SELECT BEST MODEL
print("\n" + "-"*70)
print("MODEL COMPARISON - Equipment Failure:")
for model_name, score in equipment_scores.items():
    print(f"   {model_name:20s}: ROC-AUC = {score:.4f}")

best_equip_name = max(equipment_scores, key=equipment_scores.get)
best_equip_model = equipment_models[best_equip_name]
best_equip_score = equipment_scores[best_equip_name]

print(f"\nBEST MODEL: {best_equip_name} (ROC-AUC: {best_equip_score:.4f})")
print("-"*70)

# EVALUATE BEST MODEL ON TEST SET
print("\nEvaluating best model on Test Set...")
y_test_pred = best_equip_model.predict(X_test)
y_test_prob = best_equip_model.predict_proba(X_test)[:, 1]

test_accuracy = accuracy_score(y_test_equip, y_test_pred)
test_precision = precision_score(y_test_equip, y_test_pred, zero_division=0)
test_recall = recall_score(y_test_equip, y_test_pred, zero_division=0)
test_f1 = f1_score(y_test_equip, y_test_pred, zero_division=0)
test_roc_auc = roc_auc_score(y_test_equip, y_test_prob)

print(f"   Accuracy:  {test_accuracy:.4f}")
print(f"   Precision: {test_precision:.4f}")
print(f"   Recall:    {test_recall:.4f}")
print(f"   F1-Score:  {test_f1:.4f}")
print(f"   ROC-AUC:   {test_roc_auc:.4f}")

cm = confusion_matrix(y_test_equip, y_test_pred)
print(f"\nConfusion Matrix:")
print(f"   [[TN={cm[0,0]:4d}, FP={cm[0,1]:4d}]")
print(f"    [FN={cm[1,0]:4d}, TP={cm[1,1]:4d}]]")

# SAVE BEST MODEL
os.makedirs('backend/models/saved_models', exist_ok=True)
equipment_model_path = 'backend/models/saved_models/equipment_failure_model.pkl'
joblib.dump(best_equip_model, equipment_model_path)
print(f"\nBest model saved to: {equipment_model_path}")

# STEP 4: TRAIN FLIGHT CANCELLATION MODELS
print("\n" + "="*70)
print("PART 2: FLIGHT CANCELLATION PREDICTION")
print("="*70)

X_temp, X_test_cancel, y_temp_cancel, y_test_cancel = train_test_split(
    X, y_cancellation, test_size=0.15, random_state=42, stratify=y_cancellation
)

X_train_cancel, X_val_cancel, y_train_cancel, y_val_cancel = train_test_split(
    X_temp, y_temp_cancel, test_size=0.176, random_state=42, stratify=y_temp_cancel
)

cancellation_models = {}
cancellation_scores = {}

# MODEL 1: Random Forest
print("\n[1/3] Training Random Forest...")
rf_cancel = RandomForestClassifier(
    n_estimators=100,
    max_depth=15,
    min_samples_split=5,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)
rf_cancel.fit(X_train_cancel, y_train_cancel)
y_val_prob_rf = rf_cancel.predict_proba(X_val_cancel)[:, 1]
roc_auc_rf = roc_auc_score(y_val_cancel, y_val_prob_rf)
cancellation_models['Random Forest'] = rf_cancel
cancellation_scores['Random Forest'] = roc_auc_rf
print(f"   Validation ROC-AUC: {roc_auc_rf:.4f}")

# MODEL 2: XGBoost
print("\n[2/3] Training XGBoost...")
neg_count = (y_train_cancel == 0).sum()
pos_count = (y_train_cancel == 1).sum()
scale_pos_weight = neg_count / pos_count

xgb_cancel = XGBClassifier(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    scale_pos_weight=scale_pos_weight,
    random_state=42,
    n_jobs=-1,
    eval_metric='logloss'
)
xgb_cancel.fit(X_train_cancel, y_train_cancel)
y_val_prob_xgb = xgb_cancel.predict_proba(X_val_cancel)[:, 1]
roc_auc_xgb = roc_auc_score(y_val_cancel, y_val_prob_xgb)
cancellation_models['XGBoost'] = xgb_cancel
cancellation_scores['XGBoost'] = roc_auc_xgb
print(f"   Validation ROC-AUC: {roc_auc_xgb:.4f}")

# MODEL 3: LightGBM
print("\n[3/3] Training LightGBM...")
lgbm_cancel = LGBMClassifier(
    n_estimators=100,
    max_depth=15,
    learning_rate=0.1,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1,
    verbose=-1
)
lgbm_cancel.fit(X_train_cancel, y_train_cancel)
y_val_prob_lgbm = lgbm_cancel.predict_proba(X_val_cancel)[:, 1]
roc_auc_lgbm = roc_auc_score(y_val_cancel, y_val_prob_lgbm)
cancellation_models['LightGBM'] = lgbm_cancel
cancellation_scores['LightGBM'] = roc_auc_lgbm
print(f"   Validation ROC-AUC: {roc_auc_lgbm:.4f}")

# SELECT BEST MODEL
print("\n" + "-"*70)
print("MODEL COMPARISON - Flight Cancellation:")
for model_name, score in cancellation_scores.items():
    print(f"   {model_name:20s}: ROC-AUC = {score:.4f}")

best_cancel_name = max(cancellation_scores, key=cancellation_scores.get)
best_cancel_model = cancellation_models[best_cancel_name]
best_cancel_score = cancellation_scores[best_cancel_name]

print(f"\nBEST MODEL: {best_cancel_name} (ROC-AUC: {best_cancel_score:.4f})")
print("-"*70)

# EVALUATE BEST MODEL ON TEST SET
print("\nEvaluating best model on Test Set...")
y_test_pred_cancel = best_cancel_model.predict(X_test_cancel)
y_test_prob_cancel = best_cancel_model.predict_proba(X_test_cancel)[:, 1]

test_accuracy = accuracy_score(y_test_cancel, y_test_pred_cancel)
test_precision = precision_score(y_test_cancel, y_test_pred_cancel, zero_division=0)
test_recall = recall_score(y_test_cancel, y_test_pred_cancel, zero_division=0)
test_f1 = f1_score(y_test_cancel, y_test_pred_cancel, zero_division=0)
test_roc_auc_cancel = roc_auc_score(y_test_cancel, y_test_prob_cancel)

print(f"   Accuracy:  {test_accuracy:.4f}")
print(f"   Precision: {test_precision:.4f}")
print(f"   Recall:    {test_recall:.4f}")
print(f"   F1-Score:  {test_f1:.4f}")
print(f"   ROC-AUC:   {test_roc_auc_cancel:.4f}")

cm = confusion_matrix(y_test_cancel, y_test_pred_cancel)
print(f"\nConfusion Matrix:")
print(f"   [[TN={cm[0,0]:4d}, FP={cm[0,1]:4d}]")
print(f"    [FN={cm[1,0]:4d}, TP={cm[1,1]:4d}]]")

# SAVE BEST MODEL
cancellation_model_path = 'backend/models/saved_models/flight_cancellation_model.pkl'
joblib.dump(best_cancel_model, cancellation_model_path)
print(f"\nBest model saved to: {cancellation_model_path}")

# FINAL SUMMARY
print("\n" + "="*70)
print("TRAINING COMPLETE!")
print("="*70)
print("\nFINAL RESULTS:")
print(f"Equipment Failure:     {best_equip_name} (ROC-AUC: {best_equip_score:.4f})")
print(f"Flight Cancellation:   {best_cancel_name} (ROC-AUC: {best_cancel_score:.4f})")
print("\nBest models saved and ready for deployment!")
print("="*70 + "\n")

# python backend/models/train_models.py - run this script to train models

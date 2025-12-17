"""
  ================================================================
  COMPREHENSIVE MODEL EVALUATION UTILITIES
  ================================================================
  This module provides detailed evaluation metrics for ML models,
  including both standard metrics and business-focused analysis.

  WHAT IT MEASURES:
  1. Standard ML Metrics (accuracy, precision, recall, F1, ROC-AUC)
  2. Business Metrics (cost analysis, ROI, net savings)
  3. Confusion Matrix (true positives, false positives, etc.)
  4. Visual Reports (ROC curves, confusion matrix heatmaps)

  WHY BUSINESS METRICS MATTER:
  - A model can be 95% accurate but still lose money
  - Missing a failure costs $100K (emergency repair + cancellation)
  - False alarm costs $5K (unnecessary inspection)
  - We optimize for NET SAVINGS, not just accuracy

  EXAMPLE:
  Model A: 95% accurate, misses 10 failures → LOSS $1,000,000
  Model B: 92% accurate, misses 2 failures, 50 false alarms → PROFIT $50,000
  Winner: Model B (lower accuracy, better business outcome)
  ================================================================
  """

import numpy as np
import pandas as pd
from sklearn.metrics import (
      accuracy_score, precision_score, recall_score, f1_score,
      roc_auc_score, confusion_matrix, classification_report,
      roc_curve, precision_recall_curve, average_precision_score
  )
import matplotlib.pyplot as plt
import seaborn as sns
import os


def evaluate_model_comprehensive(model, X_test, y_test, model_name, save_path=None):
      """
      Comprehensive model evaluation with business metrics

      Args:
          model: Trained sklearn model
          X_test: Test features
          y_test: Test labels (true values)
          model_name: Name of the model (for display)
          save_path: Optional path to save plots

      Returns:
          dict: Dictionary with all evaluation metrics

      METRICS CALCULATED:
      - Accuracy: % of correct predictions
      - Precision: Of predicted failures, % that are real
      - Recall: Of real failures, % that we catch (MOST IMPORTANT for safety)
      - F1-Score: Balance between precision and recall
      - ROC-AUC: Overall model quality (0.5 = random, 1.0 = perfect)
      - Business metrics: Cost analysis and net savings
      """

      # ===== MAKE PREDICTIONS =====
      y_pred = model.predict(X_test)                    # Binary predictions (0 or 1)
      y_pred_proba = model.predict_proba(X_test)[:, 1]  # Probabilities (0.0 to 1.0)

      # ===== STANDARD METRICS =====
      accuracy = accuracy_score(y_test, y_pred)
      precision = precision_score(y_test, y_pred, zero_division=0)
      recall = recall_score(y_test, y_pred, zero_division=0)
      f1 = f1_score(y_test, y_pred, zero_division=0)
      roc_auc = roc_auc_score(y_test, y_pred_proba)
      avg_precision = average_precision_score(y_test, y_pred_proba)

      # ===== CONFUSION MATRIX =====
      # Structure:
      # [[TN, FP],   TN = True Negative (correctly predicted no failure)
      #  [FN, TP]]   FP = False Positive (predicted failure, but no failure - false alarm)
      #              FN = False Negative (missed failure - DANGEROUS!)
      #              TP = True Positive (correctly caught failure)

      tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

      # ===== BUSINESS METRICS (COST ANALYSIS) =====
      # Cost assumptions (realistic industry estimates):
      # - False Negative (missed failure): $100,000
      #   → Emergency maintenance + flight cancellation + passenger compensation
      # - False Positive (unnecessary inspection): $5,000
      #   → Scheduled inspection time + labor
      # - True Positive (caught failure): SAVE $95,000
      #   → Prevented emergency by catching it early
      # - True Negative (correctly identified healthy): $0
      #   → No action needed

      fn_cost = fn * 100000      # Cost of missed failures
      fp_cost = fp * 5000        # Cost of false alarms
      savings = tp * 95000       # Savings from catching failures early
      total_cost = fn_cost + fp_cost
      net_savings = savings - fp_cost

      # ===== PRINT DETAILED REPORT =====
      print(f"\n{'='*70}")
      print(f"📊 {model_name} - COMPREHENSIVE EVALUATION")
      print(f"{'='*70}")

      # Standard metrics
      print(f"\n🎯 Standard Metrics:")
      print(f"   Accuracy:  {accuracy:.4f} (% of correct predictions)")
      print(f"   Precision: {precision:.4f} (of predicted failures, % that are real)")
      print(f"   Recall:    {recall:.4f} ⚠️ MOST CRITICAL (% of real failures we catch)")
      print(f"   F1-Score:  {f1:.4f} (balance of precision & recall)")
      print(f"   ROC-AUC:   {roc_auc:.4f} (overall quality: 0.5=random, 1.0=perfect)")
      print(f"   Avg Precision: {avg_precision:.4f}")

      # Confusion matrix breakdown
      print(f"\n📈 Confusion Matrix Breakdown:")
      print(f"   True Negatives:  {tn:,} ✅ (Correctly identified healthy aircraft)")
      print(f"   False Positives: {fp:,} ⚠️ (False alarms - unnecessary inspections)")
      print(f"   False Negatives: {fn:,} 🚨 (MISSED FAILURES - very dangerous!)")
      print(f"   True Positives:  {tp:,} ✅ (Caught failures before they happen)")

      # Business impact analysis
      print(f"\n💰 Business Impact Analysis:")
      print(f"   Cost of Missed Failures (FN): ${fn_cost:,}")
      print(f"      → {fn} failures × $100,000 each")
      print(f"   Cost of False Alarms (FP):    ${fp_cost:,}")
      print(f"      → {fp} inspections × $5,000 each")
      print(f"   Savings from Prevention (TP): ${savings:,}")
      print(f"      → {tp} failures caught × $95,000 saved each")
      print(f"   ─────────────────────────────")
      print(f"   Total Cost:                   ${total_cost:,}")
      print(f"   Net Savings:                  ${net_savings:,}")

      if net_savings > 0:
          print(f"   ✅ MODEL PROVIDES POSITIVE ROI!")
      else:
          print(f"   ⚠️ Model needs improvement for positive ROI")

      # Per-year projection (assuming dataset represents 6 months)
      annual_savings = net_savings * 2
      print(f"\n📅 Annual Projection (extrapolated):")
      print(f"   Estimated Annual Savings: ${annual_savings:,}")

      # Detailed classification report
      print(f"\n🔍 Detailed Classification Report:")
      print(classification_report(y_test, y_pred,
                                target_names=['No Failure', 'Failure'],
                                digits=4))

      # ===== RETURN METRICS DICTIONARY =====
      metrics = {
          'accuracy': accuracy,
          'precision': precision,
          'recall': recall,
          'f1': f1,
          'roc_auc': roc_auc,
          'avg_precision': avg_precision,
          'confusion_matrix': {
              'tn': int(tn),
              'fp': int(fp),
              'fn': int(fn),
              'tp': int(tp)
          },
          'business_metrics': {
              'fn_cost': int(fn_cost),
              'fp_cost': int(fp_cost),
              'savings': int(savings),
              'total_cost': int(total_cost),
              'net_savings': int(net_savings),
              'annual_savings': int(annual_savings)
          },
          'predictions': y_pred,
          'probabilities': y_pred_proba
      }

      return metrics


def plot_roc_curve(y_test, y_pred_proba, model_name, save_path=None):
      """
      Plot ROC (Receiver Operating Characteristic) curve

      ROC Curve shows trade-off between:
      - True Positive Rate (recall): % of failures we catch
      - False Positive Rate: % of healthy aircraft flagged as failures

      Perfect model: Curve goes to top-left corner (100% TPR, 0% FPR)
      Random model: Diagonal line (AUC = 0.5)

      Args:
          y_test: True labels
          y_pred_proba: Predicted probabilities
          model_name: Name for plot title
          save_path: Path to save plot image
      """
      # Calculate ROC curve points
      fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
      roc_auc = roc_auc_score(y_test, y_pred_proba)

      # Create plot
      plt.figure(figsize=(10, 6))
      plt.plot(fpr, tpr, color='darkorange', lw=2,
               label=f'ROC curve (AUC = {roc_auc:.3f})')
      plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--',
               label='Random Classifier (AUC = 0.500)')

      plt.xlim([0.0, 1.0])
      plt.ylim([0.0, 1.05])
      plt.xlabel('False Positive Rate (False Alarms)', fontsize=12)
      plt.ylabel('True Positive Rate (Recall)', fontsize=12)
      plt.title(f'{model_name} - ROC Curve', fontsize=14, fontweight='bold')
      plt.legend(loc="lower right", fontsize=11)
      plt.grid(alpha=0.3)

      # Save plot if path provided
      if save_path:
          os.makedirs(os.path.dirname(save_path), exist_ok=True)
          plt.savefig(save_path, dpi=300, bbox_inches='tight')
          print(f"   📊 ROC curve saved to: {save_path}")

      plt.close()


def plot_confusion_matrix(y_test, y_pred, model_name, save_path=None):
      """
      Plot confusion matrix as a heatmap

      Visual representation of:
      - How many predictions were correct (diagonal)
      - How many were wrong and in which way (off-diagonal)

      Args:
          y_test: True labels
          y_pred: Predicted labels
          model_name: Name for plot title
          save_path: Path to save plot image
      """
      # Calculate confusion matrix
      cm = confusion_matrix(y_test, y_pred)

      # Create heatmap
      plt.figure(figsize=(8, 6))
      sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True,
                  xticklabels=['No Failure', 'Failure'],
                  yticklabels=['No Failure', 'Failure'],
                  annot_kws={'size': 14, 'weight': 'bold'})

      plt.ylabel('True Label', fontsize=12)
      plt.xlabel('Predicted Label', fontsize=12)
      plt.title(f'{model_name} - Confusion Matrix', fontsize=14, fontweight='bold')

      # Save plot if path provided
      if save_path:
          os.makedirs(os.path.dirname(save_path), exist_ok=True)
          plt.savefig(save_path, dpi=300, bbox_inches='tight')
          print(f"   📊 Confusion matrix saved to: {save_path}")

      plt.close()


def plot_precision_recall_curve(y_test, y_pred_proba, model_name, save_path=None):
      """
      Plot Precision-Recall curve

      Shows trade-off between:
      - Precision: Of flagged aircraft, % that actually have failures
      - Recall: Of actual failures, % that we catch

      Useful for imbalanced datasets (we have ~12% failure rate)

      Args:
          y_test: True labels
          y_pred_proba: Predicted probabilities
          model_name: Name for plot title
          save_path: Path to save plot image
      """
      precision, recall, thresholds = precision_recall_curve(y_test, y_pred_proba)
      avg_precision = average_precision_score(y_test, y_pred_proba)

      plt.figure(figsize=(10, 6))
      plt.plot(recall, precision, color='blue', lw=2,
               label=f'Precision-Recall curve (AP = {avg_precision:.3f})')

      plt.xlabel('Recall (% of failures caught)', fontsize=12)
      plt.ylabel('Precision (% of alerts that are real)', fontsize=12)
      plt.title(f'{model_name} - Precision-Recall Curve', fontsize=14, fontweight='bold')
      plt.legend(loc="upper right", fontsize=11)
      plt.grid(alpha=0.3)
      plt.xlim([0.0, 1.0])
      plt.ylim([0.0, 1.05])

      if save_path:
          os.makedirs(os.path.dirname(save_path), exist_ok=True)
          plt.savefig(save_path, dpi=300, bbox_inches='tight')
          print(f"   📊 Precision-Recall curve saved to: {save_path}")

      plt.close()


  # ===== EXAMPLE USAGE (for testing) =====
if __name__ == '__main__':
      """
      Test the evaluator with dummy data
      """
      print("\n" + "="*70)
      print("TESTING MODEL EVALUATION UTILITIES")
      print("="*70)

      # Create dummy test data
      np.random.seed(42)

      # Simulate 1000 test samples
      # 120 actual failures (12% failure rate)
      y_test = np.array([0]*880 + [1]*120)

      # Simulate predictions from a "pretty good" model
      # - Catches 100 of 120 failures (83% recall)
      # - Has 40 false alarms
      y_pred = np.array(
          [0]*(880-40) +  # 840 correct no-failures
          [1]*40 +        # 40 false alarms
          [0]*20 +        # 20 missed failures
          [1]*100         # 100 caught failures
      )

      # Simulate probabilities (slightly noisy)
      y_pred_proba = y_pred.astype(float) + np.random.normal(0, 0.1, len(y_pred))
      y_pred_proba = np.clip(y_pred_proba, 0, 1)

      # Create a dummy model class
      class DummyModel:
          def predict(self, X):
              return y_pred
          def predict_proba(self, X):
              return np.column_stack([1-y_pred_proba, y_pred_proba])

      model = DummyModel()

      # Evaluate
      print("\n📊 Running comprehensive evaluation...")
      metrics = evaluate_model_comprehensive(
          model,
          None,  # X_test not needed for dummy
          y_test,
          "Test Model"
      )

      print("\n✅ Evaluation utilities test complete!")
      print("="*70)
      # python -m backend.utils.evaluator- to run
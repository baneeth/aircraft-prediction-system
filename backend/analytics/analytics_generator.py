"""
  ================================================================================
  ANALYTICS GENERATOR
  ================================================================================
  Analyzes model performance and historical data from training set
  Generates comprehensive analytics for the dashboard
  ================================================================================
  """

  import pandas as pd
  import numpy as np
  import joblib
  import json
  import os
  from sklearn.metrics import (
      roc_auc_score, accuracy_score, precision_score,
      recall_score, f1_score, confusion_matrix
  )

  # Paths
  MODEL_DIR = 'backend/models/saved_models'
  DATA_PATH = 'backend/data/processed/aircraft_flight_final.csv'
  RAW_DATA_PATH = 'backend/data/raw/aircraft_flights_data.csv'
  OUTPUT_PATH = 'backend/analytics/analytics_results.json'

  def load_models_and_data():
      """Load trained models and processed data"""
      print("\n" + "="*70)
      print("LOADING MODELS AND DATA")
      print("="*70)

      # Load models
      pipeline = joblib.load(os.path.join(MODEL_DIR, 'preprocessing_pipeline.pkl'))
      equipment_model = joblib.load(os.path.join(MODEL_DIR, 'equipment_failure_model.pkl'))
      cancellation_model = joblib.load(os.path.join(MODEL_DIR, 'flight_cancellation_model.pkl'))

      print("✓ Models loaded")

      # Load raw data (for categorical analysis)
      raw_df = pd.read_csv(RAW_DATA_PATH)
      print(f"✓ Raw data loaded: {len(raw_df):,} records")

      return pipeline, equipment_model, cancellation_model, raw_df

  def calculate_model_metrics(model, X, y_true, model_name):
      """Calculate comprehensive metrics for a model"""
      print(f"\nCalculating metrics for {model_name}...")

      # Predictions
      y_pred = model.predict(X)
      y_pred_proba = model.predict_proba(X)[:, 1]

      # Metrics
      metrics = {
          'roc_auc': float(roc_auc_score(y_true, y_pred_proba)),
          'accuracy': float(accuracy_score(y_true, y_pred)),
          'precision': float(precision_score(y_true, y_pred)),
          'recall': float(recall_score(y_true, y_pred)),
          'f1_score': float(f1_score(y_true, y_pred))
      }

      # Confusion Matrix
      cm = confusion_matrix(y_true, y_pred)
      metrics['confusion_matrix'] = {
          'true_negative': int(cm[0][0]),
          'false_positive': int(cm[0][1]),
          'false_negative': int(cm[1][0]),
          'true_positive': int(cm[1][1])
      }

      print(f"  ROC-AUC: {metrics['roc_auc']:.4f}")
      print(f"  Accuracy: {metrics['accuracy']:.4f}")
      print(f"  Precision: {metrics['precision']:.4f}")
      print(f"  Recall: {metrics['recall']:.4f}")

      return metrics

  def analyze_historical_data(raw_df):
      """Analyze historical patterns in the data"""
      print("\n" + "="*70)
      print("ANALYZING HISTORICAL DATA")
      print("="*70)

      analytics = {}

      # Overall statistics
      analytics['total_flights'] = int(len(raw_df))
      analytics['equipment_failures'] = int(raw_df['equipment_failure'].sum())
      analytics['flight_cancellations'] = int(raw_df['flight_cancelled'].sum())
      analytics['equipment_failure_rate'] = float(raw_df['equipment_failure'].mean() * 100)
      analytics['cancellation_rate'] = float(raw_df['flight_cancelled'].mean() * 100)

      print(f"\nTotal Flights: {analytics['total_flights']:,}")
      print(f"Equipment Failures: {analytics['equipment_failures']:,} ({analytics['equipment_failure_rate']:.2f}%)")
      print(f"Cancellations: {analytics['flight_cancellations']:,} ({analytics['cancellation_rate']:.2f}%)")

      # Risk distribution
      def get_risk_level(row):
          # Simplified risk calculation based on multiple factors
          risk_score = 0
          if row['days_since_maintenance'] > 300:
              risk_score += 2
          if row['aircraft_age'] > 15:
              risk_score += 1
          if row['weather_severity'] > 3:
              risk_score += 1
          if row['engine_temperature'] > 310:
              risk_score += 1

          if risk_score >= 3:
              return 'High'
          elif risk_score >= 1:
              return 'Medium'
          else:
              return 'Low'

      raw_df['risk_level'] = raw_df.apply(get_risk_level, axis=1)
      risk_distribution = raw_df['risk_level'].value_counts().to_dict()

      analytics['risk_distribution'] = {
          'low': int(risk_distribution.get('Low', 0)),
          'medium': int(risk_distribution.get('Medium', 0)),
          'high': int(risk_distribution.get('High', 0))
      }

      # Aircraft type analysis
      aircraft_analysis = raw_df.groupby('aircraft_type').agg({
          'equipment_failure': ['sum', 'mean'],
          'flight_cancelled': ['sum', 'mean']
      }).round(4)

      analytics['aircraft_types'] = []
      for aircraft_type in aircraft_analysis.index:
          analytics['aircraft_types'].append({
              'type': aircraft_type,
              'total_failures': int(aircraft_analysis.loc[aircraft_type, ('equipment_failure', 'sum')]),
              'failure_rate': float(aircraft_analysis.loc[aircraft_type, ('equipment_failure', 'mean')] * 100),
              'total_cancellations': int(aircraft_analysis.loc[aircraft_type, ('flight_cancelled', 'sum')]),
              'cancellation_rate': float(aircraft_analysis.loc[aircraft_type, ('flight_cancelled', 'mean')] * 100)
          })

      # Sort by failure rate
      analytics['aircraft_types'] = sorted(analytics['aircraft_types'],
                                          key=lambda x: x['failure_rate'],
                                          reverse=True)

      # Weather condition analysis
      weather_analysis = raw_df.groupby('weather_condition').agg({
          'equipment_failure': 'mean',
          'flight_cancelled': 'mean'
      }).round(4)

      analytics['weather_conditions'] = []
      for weather in weather_analysis.index:
          analytics['weather_conditions'].append({
              'condition': weather,
              'failure_rate': float(weather_analysis.loc[weather, 'equipment_failure'] * 100),
              'cancellation_rate': float(weather_analysis.loc[weather, 'flight_cancelled'] * 100)
          })

      # Top risky routes (by failure rate)
      route_analysis = raw_df.groupby(['origin', 'destination']).agg({
          'equipment_failure': ['sum', 'mean', 'count']
      }).round(4)

      # Filter routes with at least 50 flights
      route_analysis = route_analysis[route_analysis[('equipment_failure', 'count')] >= 50]
      route_analysis = route_analysis.sort_values(('equipment_failure', 'mean'), ascending=False)

      analytics['top_risky_routes'] = []
      for (origin, dest) in route_analysis.head(10).index:
          analytics['top_risky_routes'].append({
              'route': f"{origin} → {dest}",
              'total_flights': int(route_analysis.loc[(origin, dest), ('equipment_failure', 'count')]),
              'failures': int(route_analysis.loc[(origin, dest), ('equipment_failure', 'sum')]),
              'failure_rate': float(route_analysis.loc[(origin, dest), ('equipment_failure', 'mean')] * 100)
          })

      # Maintenance analysis
      maintenance_bins = [0, 100, 200, 300, 400, 500]
      maintenance_labels = ['0-100', '101-200', '201-300', '301-400', '401-500']
      raw_df['maintenance_category'] = pd.cut(raw_df['days_since_maintenance'],
                                              bins=maintenance_bins,
                                              labels=maintenance_labels)

      maintenance_analysis = raw_df.groupby('maintenance_category').agg({
          'equipment_failure': 'mean'
      }).round(4)

      analytics['maintenance_impact'] = []
      for category in maintenance_labels:
          if category in maintenance_analysis.index:
              analytics['maintenance_impact'].append({
                  'days_range': category,
                  'failure_rate': float(maintenance_analysis.loc[category, 'equipment_failure'] * 100)
              })

      print("\n✓ Historical analysis complete")

      return analytics

  def generate_analytics():
      """Main function to generate all analytics"""
      print("\n" + "="*70)
      print("AIRCRAFT PREDICTION SYSTEM - ANALYTICS GENERATION")
      print("="*70)

      # Load models and data
      pipeline, equipment_model, cancellation_model, raw_df = load_models_and_data()

      # Prepare data for model evaluation
      target_cols = ['equipment_failure', 'flight_cancelled']
      id_cols = ['flight_id']
      feature_cols = [col for col in raw_df.columns if col not in target_cols + id_cols]

      X = raw_df[feature_cols]
      y_equipment = raw_df['equipment_failure']
      y_cancellation = raw_df['flight_cancelled']

      # Preprocess
      X_processed = pipeline.transform(X)

      # Calculate model metrics
      print("\n" + "="*70)
      print("MODEL PERFORMANCE METRICS")
      print("="*70)

      equipment_metrics = calculate_model_metrics(
          equipment_model, X_processed, y_equipment,
          "Equipment Failure Model"
      )

      cancellation_metrics = calculate_model_metrics(
          cancellation_model, X_processed, y_cancellation,
          "Flight Cancellation Model"
      )

      # Historical data analysis
      historical_analytics = analyze_historical_data(raw_df)

      # Combine all analytics
      full_analytics = {
          'generated_at': pd.Timestamp.now().isoformat(),
          'model_performance': {
              'equipment_failure': equipment_metrics,
              'flight_cancellation': cancellation_metrics
          },
          'historical_data': historical_analytics
      }

      # Save to JSON
      print("\n" + "="*70)
      print("SAVING ANALYTICS")
      print("="*70)

      with open(OUTPUT_PATH, 'w') as f:
          json.dump(full_analytics, f, indent=2)

      print(f"✓ Analytics saved to: {OUTPUT_PATH}")
      print(f"✓ File size: {os.path.getsize(OUTPUT_PATH) / 1024:.2f} KB")

      print("\n" + "="*70)
      print("ANALYTICS GENERATION COMPLETE!")
      print("="*70 + "\n")

  if __name__ == '__main__':
      generate_analytics()
      
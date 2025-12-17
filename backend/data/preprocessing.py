"""
  ================================================================
  COMPLETE PREPROCESSING PIPELINE
  ================================================================
  This script creates a full sklearn pipeline that:
  1. Engineers features (using our custom transformer)
  2. Scales numeric features (StandardScaler)
  3. Encodes categorical features (OneHotEncoder)
  4. Handles missing values (SimpleImputer)
  5. Integrates everything into one reusable pipeline

  WHY WE NEED THIS:
  - AI models need all features as numbers
  - Features need similar scales (0-1 range)
  - Categorical data (airline, airport) needs encoding
  - Pipeline ensures consistency between training and prediction

  WHAT IT CREATES:
  - full_pipeline.pkl → Complete preprocessing pipeline
  - aircraft_flight_final.csv → Preprocessed data ready for training
  ================================================================
  """
"""
  SIMPLIFIED PREPROCESSING PIPELINE
  Works with actual columns from generate_data.py
  """

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib
import os

def create_preprocessing_pipeline():
      """
      Create preprocessing pipeline that matches our actual data columns

      Returns:
          Pipeline object
      """

      # ACTUAL NUMERIC FEATURES from generate_data.py
      numeric_features = [
          'aircraft_age',
          'operational_hours',
          'days_since_maintenance',
          'maintenance_count',
          'distance',
          'flight_duration',
          'weather_severity',
          'engine_temperature',      # Correct column name!
          'oil_pressure',
          'vibration_level',
          'fuel_consumption',
          'hydraulic_pressure',
          'cabin_pressure',
          'previous_delays',
          'crew_experience'
      ]

      # ACTUAL CATEGORICAL FEATURES from generate_data.py
      categorical_features = [
          'aircraft_type',
          'airline',
          'origin',
          'destination',
          'weather_condition'
      ]

      # Create transformers
      numeric_transformer = Pipeline([
          ('scaler', StandardScaler())  # Scale numbers to mean=0, std=1
      ])

      categorical_transformer = Pipeline([
          ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
      ])

      # Combine all transformers
      preprocessor = ColumnTransformer([
          ('num', numeric_transformer, numeric_features),
          ('cat', categorical_transformer, categorical_features)
      ], remainder='drop')

      return preprocessor, numeric_features, categorical_features


def load_and_preprocess_data(input_file):
      """Load and preprocess the data"""

      print("\n" + "="*70)
      print("DATA PREPROCESSING PIPELINE")
      print("="*70)

      # Load data
      print("\nLoading data...")
      df = pd.read_csv(input_file)
      print(f"   Loaded {len(df):,} records with {df.shape[1]} features")

      # Separate features and targets
      target_cols = ['equipment_failure', 'flight_cancelled']
      id_cols = ['flight_id']

      feature_cols = [col for col in df.columns if col not in target_cols + id_cols]

      X = df[feature_cols]
      y_equipment = df['equipment_failure']
      y_cancellation = df['flight_cancelled']

      # Print class distribution
      print(f"\nClass Distribution:")
      print(f"   Equipment Failure: {y_equipment.sum():,} / {len(y_equipment):,} ({y_equipment.mean()*100:.1f}%)")
      print(f"   Flight Cancellation: {y_cancellation.sum():,} / {len(y_cancellation):,} ({y_cancellation.mean()*100:.1f}%)")

      # Create pipeline
      print("\nCreating preprocessing pipeline...")
      pipeline, num_feats, cat_feats = create_preprocessing_pipeline()

      print(f"   Pipeline created with:")
      print(f"   - {len(num_feats)} numeric features")
      print(f"   - {len(cat_feats)} categorical features")

      # Fit and transform
      print("\nFitting and transforming data...")
      print("   (This learns statistics and applies scaling/encoding)")

      X_processed = pipeline.fit_transform(X)

      print(f"\nPreprocessing complete!")
      print(f"   Original features: {X.shape[1]}")
      print(f"   Processed features: {X_processed.shape[1]}")
      print(f"   (Increase is due to one-hot encoding of categorical features)")

      # Save pipeline
      os.makedirs('backend/models/saved_models', exist_ok=True)
      pipeline_path = 'backend/models/saved_models/preprocessing_pipeline.pkl'
      joblib.dump(pipeline, pipeline_path)

      print(f"\nPipeline saved to: {pipeline_path}")
      print(f"   This can be reused for new data predictions")

      return X_processed, y_equipment, y_cancellation, pipeline


if __name__ == '__main__':
      print("\n" + "="*70)
      print("STARTING PREPROCESSING PIPELINE")
      print("="*70)

      # Check if raw data exists
      input_path = 'backend/data/raw/aircraft_flights_data.csv'
      if not os.path.exists(input_path):
          print(f"\nERROR: Raw data not found at {input_path}")
          print("   Please run: python backend/data/generate_data.py first!")
          exit(1)

      # Preprocess data
      X, y_equip, y_cancel, pipeline = load_and_preprocess_data(input_path)

      # Save processed data
      print("\nSaving processed data...")

      output_df = pd.DataFrame(X)
      output_df['equipment_failure'] = y_equip.values
      output_df['flight_cancelled'] = y_cancel.values

      os.makedirs('backend/data/processed', exist_ok=True)
      output_path = 'backend/data/processed/aircraft_flight_final.csv'
      output_df.to_csv(output_path, index=False)

      print(f"   Processed data saved to: {output_path}")
      print(f"   Shape: {output_df.shape}")
      print(f"   Size: ~{os.path.getsize(output_path) / (1024*1024):.2f} MB")

      # Summary
      print("\n" + "="*70)
      print("PREPROCESSING COMPLETE!")
      print("="*70)
      print("\nFiles created:")
      print(f"1. Preprocessing pipeline (for reuse)")
      print(f"2. {output_path}")
      print("\nNext step: Train models using the preprocessed data!")
      print("="*70 + "\n")

#  python -m backend.data.preprocessing -to run
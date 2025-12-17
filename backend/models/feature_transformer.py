"""
  ================================================================
  FEATURE ENGINEERING TRANSFORMER
  ================================================================
  This transformer creates new features from raw aircraft data.

  WHAT IT DOES:
  - Takes raw data (engine temp, weather, etc.)
  - Creates 30+ engineered features
  - Integrates with scikit-learn Pipeline
  - Learns from training data, applies to test data

  WHY WE NEED IT:
  Raw data alone isn't enough for AI models. Engineered features
  help models find patterns more easily.

  EXAMPLE:
  Raw data:        engine_temp=310, vibration=16
  Engineered:      engine_stress_index = 310 × 16 / 1000 = 4.96
  AI learns:       "High stress index → Equipment failure"

  FEATURES CREATED:
  1. Time-based (weekend, holiday season, winter, rush hour)
  2. Sensor health (temp/pressure ratio, stress index, anomalies)
  3. Weather risk (combined origin + destination risk scores)
  4. Aircraft health (composite health score, cycle stress)
  5. Interactions (age × hours, weather × health)
  6. Binary flags (overdue maintenance, old aircraft, etc.)
  ================================================================
  """

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class FeatureEngineeringTransformer(BaseEstimator, TransformerMixin):
      """
      Custom sklearn transformer for aircraft feature engineering

      INHERITANCE:
      - BaseEstimator: Provides get_params() and set_params() for sklearn
      - TransformerMixin: Provides fit_transform() method

      HOW IT WORKS:
      1. fit(X, y): Learn statistics from training data (e.g., mean vibration)
      2. transform(X): Apply feature engineering using learned statistics
      3. fit_transform(X, y): Fit and transform in one step (for training)
      """

      def __init__(self):
          """
          Initialize the transformer

          We'll store statistics here during fit() to use during transform()
          This ensures consistency between training and test data
          """
          # These will be set during fit()
          self.vibration_mean_ = None
          self.vibration_std_ = None

      def fit(self, X, y=None):
          """
          Learn statistics from the training data

          Args:
              X: Training data (DataFrame or array)
              y: Target variable (not used here, but sklearn requires it)

          Returns:
              self: Returns the transformer itself (sklearn convention)

          WHAT HAPPENS:
          - Calculates mean and std of vibration_level from training data
          - Stores these for use in transform()
          - This prevents data leakage (using test data info during training)
          """
          # Convert to DataFrame if needed
          if isinstance(X, pd.DataFrame):
              df = X.copy()
          else:
              df = pd.DataFrame(X)

          # Learn statistics for standardization-based features
          if 'vibration_level' in df.columns:
              self.vibration_mean_ = df['vibration_level'].mean()
              self.vibration_std_ = df['vibration_level'].std()

          return self  # Always return self in fit()

      def transform(self, X):
          """
          Apply feature engineering to data

          Args:
              X: Data to transform (DataFrame or array)

          Returns:
              DataFrame with original + engineered features

          PROCESS:
          1. Create time-based features (weekend, season, etc.)
          2. Create sensor health features (ratios, indices)
          3. Create weather risk scores
          4. Create aircraft health metrics
          5. Create interaction features
          6. Create composite risk scores
          """
          # Convert to DataFrame if needed
          if isinstance(X, pd.DataFrame):
              df = X.copy()
          else:
              df = pd.DataFrame(X)

          # ========================================
          # 1. TIME-BASED FEATURES
          # ========================================
          # These help capture seasonal patterns

          if 'day_of_week' in df.columns:
              # Weekend flights might have different failure patterns
              df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)

          if 'month' in df.columns:
              # Holiday seasons = more flights = more stress
              df['is_holiday_season'] = df['month'].isin([11, 12, 6, 7]).astype(int)
              # Winter = harsher conditions
              df['is_winter'] = df['month'].isin([12, 1, 2]).astype(int)

          if 'departure_hour' in df.columns:
              # Rush hour flights = tighter schedules = less maintenance time
              df['is_rush_hour'] = df['departure_hour'].isin([6, 7, 8, 17, 18, 19]).astype(int)

          # ========================================
          # 2. SENSOR HEALTH FEATURES
          # ========================================
          # These combine multiple sensors to detect problems

          if 'engine_temp' in df.columns and 'oil_pressure' in df.columns:
              # High temp with low pressure = BAD (lubrication problem)
              # Adding 1 to avoid division by zero
              df['temp_pressure_ratio'] = df['engine_temp'] / (df['oil_pressure'] + 1)

          if 'vibration_level' in df.columns and self.vibration_mean_ is not None:
              # How many standard deviations away from normal?
              # High positive value = abnormal vibration
              df['vibration_anomaly'] = (
                  (df['vibration_level'] - self.vibration_mean_) /
                  (self.vibration_std_ + 1e-8)  # Add tiny number to avoid division by zero
              )

          if 'engine_temp' in df.columns and 'vibration_level' in df.columns:
              # Combined stress: Hot engine + High vibration = DANGER
              df['engine_stress_index'] = (
                  df['engine_temp'] * df['vibration_level']
              ) / 1000  # Divide by 1000 to keep numbers reasonable

          if 'operational_hours' in df.columns:
              # Non-linear relationship: More hours = exponentially more wear
              df['operational_hours_squared'] = df['operational_hours'] ** 2
              # Log transform: Compresses large values
              df['operational_hours_log'] = np.log1p(df['operational_hours'])

          if 'days_since_last_maintenance' in df.columns:
              # Binary flags are easier for models to learn
              df['maintenance_overdue'] = (
                  df['days_since_last_maintenance'] > 90
              ).astype(int)
              # Critical = REALLY overdue
              df['maintenance_critical'] = (
                  df['days_since_last_maintenance'] > 150
              ).astype(int)

          # ========================================
          # 3. WEATHER RISK SCORES
          # ========================================
          # Combine multiple weather factors into a single "risk" number

          if all(col in df.columns for col in ['origin_precipitation', 'origin_wind_speed', 'origin_visibility']):
              # Higher score = worse weather
              df['origin_weather_risk'] = (
                  (df['origin_precipitation'] * 2) +              # Rain/snow: +2
                  (df['origin_wind_speed'] > 30).astype(int) * 2 + # High wind: +2
                  (df['origin_visibility'] < 5).astype(int) * 3 +  # Low visibility: +3
                  (df['origin_wind_speed'] > 40).astype(int) * 2   # Very high wind: +2 more
              )

          if all(col in df.columns for col in ['destination_precipitation', 'destination_wind_speed', 'destination_visibility']):
              df['destination_weather_risk'] = (
                  (df['destination_precipitation'] * 2) +
                  (df['destination_wind_speed'] > 30).astype(int) * 2 +
                  (df['destination_visibility'] < 5).astype(int) * 3 +
                  (df['destination_wind_speed'] > 40).astype(int) * 2
              )

          if 'origin_weather_risk' in df.columns and 'destination_weather_risk' in df.columns:
              # Total route weather risk
              df['total_weather_risk'] = (
                  df['origin_weather_risk'] + df['destination_weather_risk']
              )
              # Binary: Is this a severe weather route?
              df['severe_weather_route'] = (
                  df['total_weather_risk'] > 8
              ).astype(int)

          if 'origin_temperature' in df.columns and 'destination_temperature' in df.columns:
              # Large temp change = stress on aircraft materials
              df['temp_differential'] = abs(
                  df['origin_temperature'] - df['destination_temperature']
              )

          # ========================================
          # 4. AIRCRAFT HEALTH FEATURES
          # ========================================
          # Overall health score combining multiple factors

          if all(col in df.columns for col in ['aircraft_age', 'avionics_health_score', 'days_since_last_maintenance']):
              # Weighted average of health factors (0-100 scale)
              df['aircraft_health_score'] = (
                  (100 - df['aircraft_age'] * 2).clip(0, 100) * 0.3 +  # 30% weight
                  df['avionics_health_score'] * 0.4 +                    # 40% weight
                  (100 - df['days_since_last_maintenance'] / 2).clip(0, 100) * 0.3  # 30% weight
              )

          if 'takeoff_landing_cycles' in df.columns:
              # High cycles = more stress (each takeoff/landing is hard on aircraft)
              df['high_cycle_aircraft'] = (
                  df['takeoff_landing_cycles'] > 3000
              ).astype(int)
              # Log transform to compress range
              df['cycle_stress_factor'] = np.log1p(df['takeoff_landing_cycles'])

          if 'aircraft_age' in df.columns:
              # Non-linear aging: Old aircraft degrade faster
              df['aircraft_age_squared'] = df['aircraft_age'] ** 2
              df['old_aircraft'] = (df['aircraft_age'] > 15).astype(int)

          # ========================================
          # 5. INTERACTION FEATURES
          # ========================================
          # Combinations that capture relationships between features

          if 'aircraft_age' in df.columns and 'operational_hours' in df.columns:
              # Old aircraft with many hours = highest risk
              df['age_hours_interaction'] = (
                  df['aircraft_age'] * df['operational_hours']
              ) / 1000  # Scale down

          if 'total_weather_risk' in df.columns and 'aircraft_health_score' in df.columns:
              # Bad weather + unhealthy aircraft = compounded risk
              df['weather_health_risk'] = (
                  df['total_weather_risk'] * (100 - df['aircraft_health_score'])
              ) / 100  # Normalize

          if 'previous_cancellations_count' in df.columns:
              # History of cancellations = unreliable aircraft/route
              df['cancellation_history_flag'] = (
                  df['previous_cancellations_count'] > 2
              ).astype(int)

          if 'previous_maintenance_issues' in df.columns:
              # Frequent issues = problematic aircraft
              df['frequent_maintenance_issues'] = (
                  df['previous_maintenance_issues'] > 4
              ).astype(int)

          # ========================================
          # 6. FLIGHT COMPLEXITY FEATURES
          # ========================================

          if 'distance' in df.columns and 'flight_duration' in df.columns:
              # Complex flights = higher risk
              df['flight_complexity_score'] = (
                  (df['distance'] / 1000) * 0.4 +          # Long distance
                  (df['flight_duration'] / 100) * 0.3      # Long duration
              )

              if 'total_weather_risk' in df.columns:
                  # Add weather to complexity
                  df['flight_complexity_score'] += df['total_weather_risk'] * 0.3

          if 'distance' in df.columns:
              # Long haul flights stress aircraft more
              df['long_haul_flight'] = (df['distance'] > 2000).astype(int)

          # ========================================
          # 7. CREW & OPERATIONAL FEATURES
          # ========================================

          if 'crew_experience' in df.columns:
              # Inexperienced crew = higher operational risk
              df['inexperienced_crew'] = (df['crew_experience'] < 2000).astype(int)
              df['crew_experience_log'] = np.log1p(df['crew_experience'])

          # ========================================
          # 8. COMPOSITE RISK SCORES
          # ========================================
          # Combine multiple engineered features into final risk metrics

          # Equipment risk composite
          equipment_risk_features = []
          if 'engine_stress_index' in df.columns:
              equipment_risk_features.append(df['engine_stress_index'])
          if 'maintenance_overdue' in df.columns:
              equipment_risk_features.append(df['maintenance_overdue'] * 10)
          if 'old_aircraft' in df.columns:
              equipment_risk_features.append(df['old_aircraft'] * 8)
          if 'vibration_anomaly' in df.columns:
              equipment_risk_features.append(df['vibration_anomaly'].clip(0, 10))

          if equipment_risk_features:
              # Average of all equipment risk factors
              df['composite_equipment_risk'] = sum(equipment_risk_features) / len(equipment_risk_features)

          # Operational risk composite
          operational_risk_features = []
          if 'total_weather_risk' in df.columns:
              operational_risk_features.append(df['total_weather_risk'])
          if 'flight_complexity_score' in df.columns:
              operational_risk_features.append(df['flight_complexity_score'])
          if 'cancellation_history_flag' in df.columns:
              operational_risk_features.append(df['cancellation_history_flag'] * 5)

          if operational_risk_features:
              df['composite_operational_risk'] = sum(operational_risk_features) / len(operational_risk_features)

          return df

      def get_feature_names_out(self, input_features=None):
          """
          Get names of all output features (original + engineered)

          This method is required for sklearn pipeline compatibility

          Args:
              input_features: List of input feature names

          Returns:
              List of all feature names (input + engineered)
          """
          if input_features is None:
              return None

          # List of all engineered features we create
          engineered_features = [
              # Time-based
              'is_weekend', 'is_holiday_season', 'is_winter', 'is_rush_hour',

              # Sensor health
              'temp_pressure_ratio', 'vibration_anomaly', 'engine_stress_index',
              'operational_hours_squared', 'operational_hours_log',
              'maintenance_overdue', 'maintenance_critical',

              # Weather risk
              'origin_weather_risk', 'destination_weather_risk', 'total_weather_risk',
              'severe_weather_route', 'temp_differential',

              # Aircraft health
              'aircraft_health_score', 'high_cycle_aircraft', 'cycle_stress_factor',
              'aircraft_age_squared', 'old_aircraft',

              # Interactions
              'age_hours_interaction', 'weather_health_risk',
              'cancellation_history_flag', 'frequent_maintenance_issues',

              # Flight complexity
              'flight_complexity_score', 'long_haul_flight',

              # Crew
              'inexperienced_crew', 'crew_experience_log',

              # Composite risks
              'composite_equipment_risk', 'composite_operational_risk'
          ]

          # Return original + engineered features
          return list(input_features) + engineered_features


  # ===== EXAMPLE USAGE (for testing) =====
if __name__ == '__main__':
      """
      Test the feature transformer with sample data
      """
      print("="*70)
      print("TESTING FEATURE ENGINEERING TRANSFORMER")
      print("="*70)

      # Create sample data (3 rows)
      sample_data = pd.DataFrame({
          'engine_temp': [285, 310, 275],
          'oil_pressure': [45, 35, 50],
          'vibration_level': [10, 16, 8],
          'operational_hours': [5000, 8500, 2000],
          'aircraft_age': [10, 18, 5],
          'days_since_last_maintenance': [60, 155, 30],
          'takeoff_landing_cycles': [2500, 4200, 1000],
          'month': [12, 6, 3],
          'day_of_week': [5, 2, 1],
          'departure_hour': [8, 14, 20],
          'origin_precipitation': [1, 0, 0],
          'origin_wind_speed': [35, 15, 20],
          'origin_visibility': [4, 10, 8],
          'destination_precipitation': [0, 1, 0],
          'destination_wind_speed': [20, 25, 15],
          'destination_visibility': [8, 6, 10],
          'origin_temperature': [30, 85, 60],
          'destination_temperature': [75, 80, 55],
          'avionics_health_score': [85, 70, 95],
          'previous_cancellations_count': [1, 4, 0],
          'previous_maintenance_issues': [2, 6, 1],
          'crew_experience': [5000, 1500, 8000],
          'distance': [1500, 2500, 800],
          'flight_duration': [180, 300, 120]
      })

      print("\n📊 Original Data Shape:", sample_data.shape)
      print("Original Features:", list(sample_data.columns))

      # Create and fit transformer
      transformer = FeatureEngineeringTransformer()
      transformer.fit(sample_data)

      print(f"\n🔧 Learned Statistics:")
      print(f"   Vibration Mean: {transformer.vibration_mean_:.2f}")
      print(f"   Vibration Std: {transformer.vibration_std_:.2f}")

      # Transform data
      transformed_data = transformer.transform(sample_data)

      print(f"\n✨ Transformed Data Shape:", transformed_data.shape)
      print(f"   Original Features: {sample_data.shape[1]}")
      print(f"   Engineered Features: {transformed_data.shape[1] - sample_data.shape[1]}")
      print(f"   Total Features: {transformed_data.shape[1]}")

      print(f"\n📋 Sample Engineered Features:")
      engineered_cols = [col for col in transformed_data.columns if col not in sample_data.columns]
      print(f"   {engineered_cols[:10]}...")

      print(f"\n✅ Feature Engineering Transformer Test Complete!")
      print("="*70)
#python backend/models/feature_transformer.py -to run
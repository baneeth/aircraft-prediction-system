"""
  Debug script to investigate data issues
  """

import pandas as pd
import numpy as np
import os

def check_raw_data():
      """Check the original generated data"""
      print("\n" + "="*70)
      print("🔍 CHECKING RAW DATA")
      print("="*70)

      raw_path = 'backend/data/raw/aircraft_flight_data.csv'

      if not os.path.exists(raw_path):
          print("❌ Raw data not found!")
          return None

      df = pd.read_csv(raw_path)

      print(f"\n📊 Raw Data Statistics:")
      print(f"   Total records: {len(df):,}")
      print(f"   Total columns: {df.shape[1]}")

      print(f"\n🔧 Equipment Failure Distribution:")
      print(f"   Failures (1): {df['equipment_failure'].sum():,} ({df['equipment_failure'].mean()*100:.2f}%)")
      print(f"   Normal (0):   {(df['equipment_failure']==0).sum():,} ({(df['equipment_failure']==0).mean()*100:.2f}%)")

      print(f"\n✈️ Flight Cancellation Distribution:")
      print(f"   Cancelled (1): {df['flight_cancelled'].sum():,} ({df['flight_cancelled'].mean()*100:.2f}%)")
      print(f"   Completed (0): {(df['flight_cancelled']==0).sum():,} ({(df['flight_cancelled']==0).mean()*100:.2f}%)")

      print(f"\n🎯 Cancellation Reasons:")
      print(df[df['flight_cancelled']==1]['cancellation_reason'].value_counts())

      print(f"\n📋 First 5 rows of target variables:")
      print(df[['flight_id', 'equipment_failure', 'flight_cancelled', 'cancellation_reason']].head())

      return df


def check_processed_data():
      """Check the preprocessed data"""
      print("\n" + "="*70)
      print("🔍 CHECKING PROCESSED DATA")
      print("="*70)

      processed_path = 'backend/data/processed/aircraft_flight_final.csv'

      if not os.path.exists(processed_path):
          print("❌ Processed data not found!")
          return None

      df = pd.read_csv(processed_path)

      print(f"\n📊 Processed Data Statistics:")
      print(f"   Total records: {len(df):,}")
      print(f"   Total columns: {df.shape[1]}")

      print(f"\n🔧 Equipment Failure Distribution:")
      print(f"   Failures (1): {df['equipment_failure'].sum():,} ({df['equipment_failure'].mean()*100:.2f}%)")
      print(f"   Normal (0):   {(df['equipment_failure']==0).sum():,} ({(df['equipment_failure']==0).mean()*100:.2f}%)")

      print(f"\n✈️ Flight Cancellation Distribution:")
      print(f"   Cancelled (1): {df['flight_cancelled'].sum():,} ({df['flight_cancelled'].mean()*100:.2f}%)")
      print(f"   Completed (0): {(df['flight_cancelled']==0).sum():,} ({(df['flight_cancelled']==0).mean()*100:.2f}%)")

      print(f"\n📋 Sample of processed features (first 5 columns):")
      print(df.iloc[:5, :5])

      print(f"\n📋 Last 3 columns (should be targets):")
      print(df.iloc[:5, -3:])

      return df


def check_feature_values(df_raw):
      """Check if feature values make sense"""
      print("\n" + "="*70)
      print("🔍 CHECKING FEATURE VALUE RANGES")
      print("="*70)

      print(f"\n📊 Sensor Data Ranges:")
      print(f"   Engine Temp: {df_raw['engine_temp'].min():.1f} - {df_raw['engine_temp'].max():.1f}°C (normal: 260-320)")
      print(f"   Oil Pressure: {df_raw['oil_pressure'].min():.1f} - {df_raw['oil_pressure'].max():.1f} PSI (normal: 30-60)")
      print(f"   Vibration: {df_raw['vibration_level'].min():.1f} - {df_raw['vibration_level'].max():.1f} mm/s (normal: 5-20)")

      print(f"\n📊 Aircraft Data:")
      print(f"   Aircraft Age: {df_raw['aircraft_age'].min()} - {df_raw['aircraft_age'].max()} years")
      print(f"   Days Since Maintenance: {df_raw['days_since_last_maintenance'].min()} - {df_raw['days_since_last_maintenance'].max()} days")

      print(f"\n📊 Correlation with Failures:")

      # Check if high-risk conditions correlate with failures
      high_temp = df_raw['engine_temp'] > 300
      high_vibration = df_raw['vibration_level'] > 15
      overdue_maintenance = df_raw['days_since_last_maintenance'] > 120

      print(f"\n   Aircraft with high temp (>300°C): {high_temp.sum()} ({high_temp.mean()*100:.1f}%)")
      print(f"      Of these, {df_raw[high_temp]['equipment_failure'].mean()*100:.1f}% have failures")

      print(f"\n   Aircraft with high vibration (>15): {high_vibration.sum()} ({high_vibration.mean()*100:.1f}%)")
      print(f"      Of these, {df_raw[high_vibration]['equipment_failure'].mean()*100:.1f}% have failures")

      print(f"\n   Aircraft with overdue maintenance (>120 days): {overdue_maintenance.sum()} ({overdue_maintenance.mean()*100:.1f}%)")
      print(f"      Of these, {df_raw[overdue_maintenance]['equipment_failure'].mean()*100:.1f}% have failures")


def compare_data():
      """Compare raw vs processed to find where things changed"""
      print("\n" + "="*70)
      print("🔍 COMPARING RAW VS PROCESSED DATA")
      print("="*70)

      raw_path = 'backend/data/raw/aircraft_flight_data.csv'
      processed_path = 'backend/data/processed/aircraft_flight_final.csv'

      if not os.path.exists(raw_path) or not os.path.exists(processed_path):
          print("❌ Cannot compare - missing files")
          return

      df_raw = pd.read_csv(raw_path)
      df_processed = pd.read_csv(processed_path)

      print(f"\n📊 Record Count Comparison:")
      print(f"   Raw:       {len(df_raw):,} records")
      print(f"   Processed: {len(df_processed):,} records")

      if len(df_raw) == len(df_processed):
          print("   ✅ Same number of records")
      else:
          print(f"   ⚠️ Different! Missing {abs(len(df_raw) - len(df_processed))} records")

      print(f"\n🔧 Equipment Failure Comparison:")
      raw_failures = df_raw['equipment_failure'].sum()
      processed_failures = df_processed['equipment_failure'].sum()

      print(f"   Raw:       {raw_failures:,} ({df_raw['equipment_failure'].mean()*100:.2f}%)")
      print(f"   Processed: {processed_failures:,} ({df_processed['equipment_failure'].mean()*100:.2f}%)")

      if raw_failures == processed_failures:
          print("   ✅ Same number of failures")
      else:
          print(f"   ❌ DIFFERENT! Difference: {abs(raw_failures - processed_failures)}")
          print("   🐛 BUG FOUND: Preprocessing changed the labels!")

      print(f"\n✈️ Flight Cancellation Comparison:")
      raw_cancellations = df_raw['flight_cancelled'].sum()
      processed_cancellations = df_processed['flight_cancelled'].sum()

      print(f"   Raw:       {raw_cancellations:,} ({df_raw['flight_cancelled'].mean()*100:.2f}%)")
      print(f"   Processed: {processed_cancellations:,} ({df_processed['flight_cancelled'].mean()*100:.2f}%)")

      if raw_cancellations == processed_cancellations:
          print("   ✅ Same number of cancellations")
      else:
          print(f"   ❌ DIFFERENT! Difference: {abs(raw_cancellations - processed_cancellations)}")
          print("   🐛 BUG FOUND: Preprocessing changed the labels!")


def main():
      """Run all debug checks"""
      print("\n" + "="*70)
      print("🐛 DATA DEBUGGING TOOL")
      print("="*70)
      print("\nThis will help us find what went wrong with the training data.")

      # Check raw data
      df_raw = check_raw_data()

      # Check processed data
      df_processed = check_processed_data()

      # Check feature values
      if df_raw is not None:
          check_feature_values(df_raw)

      # Compare raw vs processed
      compare_data()

      print("\n" + "="*70)
      print("🔍 DEBUGGING COMPLETE")
      print("="*70)
      print("\nLook for:")
      print("❌ Abnormal failure rates in raw data (should be ~12%, not 60%)")
      print("❌ Changes between raw and processed data")
      print("❌ Feature values outside normal ranges")
      print("="*70 + "\n")


if __name__ == '__main__':
      main()
      #python debug_data.py -to run
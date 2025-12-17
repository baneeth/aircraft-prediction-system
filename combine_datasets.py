"""
================================================================================
DATASET COMBINER - Merge Main + Incremental Data (with Auto-Archive)
================================================================================
This script:
1. Combines main dataset + all incremental datasets
2. Saves combined dataset
3. ARCHIVES used incremental files (moves them to 'archived' folder)
4. Next run will only combine NEW incremental data

This prevents duplicates!
================================================================================
"""

import pandas as pd
import glob
import os
import shutil
from datetime import datetime

print("\n" + "="*70)
print("DATASET COMBINER (with Auto-Archive)")
print("="*70)

# STEP 1: Load main dataset
print("\n[1/5] Loading main dataset...")
main_data_path = 'backend/data/raw/aircraft_flights_data.csv'

if not os.path.exists(main_data_path):
    print(f"ERROR: Main dataset not found at {main_data_path}")
    exit(1)

main_df = pd.read_csv(main_data_path)
print(f"   Main dataset: {len(main_df):,} records")
print(f"   Equipment failure rate: {main_df['equipment_failure'].mean()*100:.2f}%")
print(f"   Cancellation rate: {main_df['flight_cancelled'].mean()*100:.2f}%")

# STEP 2: Find all incremental datasets
print("\n[2/5] Finding incremental datasets...")
incremental_files = glob.glob('backend/data/raw/incremental_data_*.csv')

if len(incremental_files) == 0:
    print("   No incremental datasets found")
    print("   Nothing to combine. Main dataset unchanged.")
    exit(0)

print(f"   Found {len(incremental_files)} incremental dataset(s)")

# Load all incremental datasets
incremental_dfs = []
total_incremental_records = 0

for file_path in incremental_files:
    filename = os.path.basename(file_path)
    df = pd.read_csv(file_path)
    incremental_dfs.append(df)
    total_incremental_records += len(df)
    print(f"      - {filename}: {len(df):,} records")

print(f"   Total incremental records: {total_incremental_records:,}")

# STEP 3: Combine all datasets
print("\n[3/5] Combining datasets...")
all_dfs = [main_df] + incremental_dfs
combined_df = pd.concat(all_dfs, ignore_index=True)
print(f"   Total combined records: {len(combined_df):,}")

# Remove duplicates based on flight_id if it exists
if 'flight_id' in combined_df.columns:
    before_dedup = len(combined_df)
    combined_df = combined_df.drop_duplicates(subset=['flight_id'], keep='first')
    duplicates_removed = before_dedup - len(combined_df)
    if duplicates_removed > 0:
        print(f"   Removed {duplicates_removed} duplicate flight IDs")
    print(f"   Final dataset: {len(combined_df):,} unique records")

# STEP 4: Save combined dataset (REPLACE the main dataset)
print("\n[4/5] Updating main dataset...")

# Backup original main dataset (first time only)
backup_path = 'backend/data/raw/aircraft_flights_data_ORIGINAL_BACKUP.csv'
if not os.path.exists(backup_path):
    shutil.copy(main_data_path, backup_path)
    print(f"   Created backup: {os.path.basename(backup_path)}")

# Replace main dataset with combined data
combined_df.to_csv(main_data_path, index=False)
file_size_mb = os.path.getsize(main_data_path) / (1024 * 1024)
print(f"   Updated: {main_data_path}")
print(f"   File size: {file_size_mb:.2f} MB")

# STEP 5: Archive incremental files
print("\n[5/5] Archiving incremental datasets...")

# Create archived folder
archive_folder = 'backend/data/raw/archived'
os.makedirs(archive_folder, exist_ok=True)

archived_count = 0
for file_path in incremental_files:
    filename = os.path.basename(file_path)
    archive_path = os.path.join(archive_folder, filename)
    shutil.move(file_path, archive_path)
    archived_count += 1
    print(f"   Archived: {filename}")

print(f"   Moved {archived_count} file(s) to: {archive_folder}")

# STEP 6: Show final statistics
print("\n" + "="*70)
print("UPDATED MAIN DATASET STATISTICS")
print("="*70)
print(f"Total Records:           {len(combined_df):,}")
print(f"Equipment Failure Rate:  {combined_df['equipment_failure'].mean()*100:.2f}%")
print(f"   - Failures:           {combined_df['equipment_failure'].sum():,}")
print(f"   - Normal:             {(~combined_df['equipment_failure'].astype(bool)).sum():,}")
print(f"\nFlight Cancellation Rate: {combined_df['flight_cancelled'].mean()*100:.2f}%")
print(f"   - Cancelled:          {combined_df['flight_cancelled'].sum():,}")
print(f"   - Completed:          {(~combined_df['flight_cancelled'].astype(bool)).sum():,}")

if 'aircraft_type' in combined_df.columns:
    print(f"\nAircraft Types:")
    for aircraft, count in combined_df['aircraft_type'].value_counts().head(5).items():
        print(f"   - {aircraft}: {count:,} flights")

print("\n" + "="*70)
print("DATASET COMBINATION COMPLETE!")
print("="*70)
print("\nWhat happened:")
print("   1. Combined main dataset + incremental datasets")
print("   2. Updated aircraft_flights_data.csv with combined data")
print("   3. Moved incremental files to 'archived' folder")
print("   4. Next run will only combine NEW incremental data (no duplicates!)")
print("\nNEXT STEPS:")
print("   1. Run: python backend/data/preprocessing.py")
print("   2. Run: python backend/models/train_models.py")
print("="*70 + "\n")
# python combine_datasets.py-to run
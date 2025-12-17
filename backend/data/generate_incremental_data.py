"""
  ================================================================
  INCREMENTAL DATA GENERATOR - MULTI-FILE MODE
  ================================================================
  This script generates NEW incremental datasets each time you run it.
  - Creates TIMESTAMPED files (never overwrites)
  - Generates 3,000 records by default (easily changeable)
  - Different random seed each time (simulates real-world data collection)
  - Slightly different patterns (data drift simulation)

  USE CASES:
  1. Generate multiple datasets for different time periods
  2. Test model retraining with new data
  3. Simulate continuous learning scenarios

  FILES CREATED:
  - incremental_data_2024_01_15_143022.csv
  - incremental_data_2024_01_16_091545.csv
  - incremental_data_2024_01_20_162033.csv
  (etc.)
  ================================================================
  """

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os


class IncrementalDataGenerator:
      """
      Generates incremental flight data with timestamp-based filenames
      Each run creates a NEW file with different random data
      """

      def __init__(self, use_timestamp_seed=True):
          """
          Initialize the incremental data generator

          Args:
              use_timestamp_seed: If True, uses current time as random seed
                                (makes each run different)
          """
          # Use current timestamp as seed for different data each time
          if use_timestamp_seed:
              seed = int(datetime.now().timestamp())
              print(f"🔀 Using timestamp-based seed: {seed}")
              print("   (This ensures different data each time you run)")
          else:
              seed = 42  # Fixed seed for reproducibility
              print("🔒 Using fixed seed: 42 (data will be identical each run)")

          np.random.seed(seed)

          # ===== REFERENCE DATA (same as main generator) =====
          self.airlines = ['AA', 'UA', 'DL', 'WN', 'B6', 'AS', 'NK', 'F9']

          self.airports = [
              'JFK', 'LAX', 'ORD', 'DFW', 'ATL', 'DEN', 'SFO', 'SEA',
              'LAS', 'MCO', 'MIA', 'BOS', 'EWR', 'PHX', 'IAH', 'CLT'
          ]

          self.aircraft_types = {
              'Boeing 737': {'age_range': (0, 25), 'reliability': 85},
              'Airbus A320': {'age_range': (0, 20), 'reliability': 88},
              'Boeing 777': {'age_range': (5, 22), 'reliability': 90},
              'Airbus A330': {'age_range': (3, 20), 'reliability': 87},
              'Boeing 787': {'age_range': (0, 12), 'reliability': 92},
              'Embraer E175': {'age_range': (0, 15), 'reliability': 82}
          }

          self.weather_conditions = ['Clear', 'Cloudy', 'Rainy', 'Snowy', 'Foggy']

          self.airport_coords = {
              'JFK': (40.6413, -73.7781), 'LAX': (33.9416, -118.4085),
              'ORD': (41.9742, -87.9073), 'DFW': (32.8998, -97.0403),
              'ATL': (33.6407, -84.4277), 'DEN': (39.8561, -104.6737),
              'SFO': (37.6213, -122.3790), 'SEA': (47.4502, -122.3088),
              'LAS': (36.0840, -115.1537), 'MCO': (28.4312, -81.3081),
              'MIA': (25.7959, -80.2870), 'BOS': (42.3656, -71.0096),
              'EWR': (40.6895, -74.1745), 'PHX': (33.4352, -112.0101),
              'IAH': (29.9902, -95.3368), 'CLT': (35.2144, -80.9473)
          }

      def calculate_distance(self, origin, destination):
          """Calculate flight distance using Haversine formula"""
          if origin == destination:
              return 0

          lat1, lon1 = self.airport_coords.get(origin, (0, 0))
          lat2, lon2 = self.airport_coords.get(destination, (0, 0))

          R = 3959  # Earth radius in miles
          lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
          dlat = lat2 - lat1
          dlon = lon2 - lon1
          a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
          c = 2 * np.arcsin(np.sqrt(a))

          return int(R * c)

      def estimate_duration(self, distance):
          """Estimate flight duration in minutes"""
          avg_speed = 500
          flight_time = (distance / avg_speed) * 60
          return int(flight_time + np.random.randint(20, 40))

      def generate_sensor_data(self, aircraft_age, operational_hours, days_since_maintenance):
          """
          Generate sensor data with slight variation from main dataset
          (Simulates evolving sensor patterns over time)
          """
          base_engine_temp = 280
          base_oil_pressure = 45
          base_vibration = 10

          # Slightly increased degradation (simulates aging fleet)
          age_factor = 1 + (aircraft_age / 95)  # Was 100, now 95 (slightly worse)
          hours_factor = 1 + (operational_hours / 19000)  # Was 20000
          maintenance_factor = 1 + (days_since_maintenance / 480)  # Was 500

          engine_temp = np.random.normal(
              base_engine_temp * age_factor * maintenance_factor,
              20
          )

          oil_pressure = np.random.normal(
              base_oil_pressure / maintenance_factor,
              8
          )

          vibration_level = np.random.normal(
              base_vibration * hours_factor * age_factor,
              3
          )

          engine_rpm = np.random.normal(2500, 200)
          fuel_consumption = np.random.normal(3000, 300)
          hydraulic_pressure = np.random.normal(3000, 200)
          tire_pressure = np.random.normal(200, 15)
          brake_temp = np.random.normal(150, 30)

          avionics_health = int(np.clip(
              np.random.normal(95 - aircraft_age, 10),
              60, 100
          ))

          return {
              'engine_temp': round(engine_temp, 1),
              'oil_pressure': round(oil_pressure, 1),
              'vibration_level': round(vibration_level, 2),
              'engine_rpm': int(engine_rpm),
              'fuel_consumption_rate': int(fuel_consumption),
              'hydraulic_pressure': int(hydraulic_pressure),
              'tire_pressure': int(tire_pressure),
              'brake_temperature': int(brake_temp),
              'avionics_health_score': avionics_health
          }

      def generate_weather(self, airport, month):
          """Generate weather conditions"""
          is_winter = month in [12, 1, 2]
          is_summer = month in [6, 7, 8]

          if is_winter:
              temp = np.random.randint(25, 50)
              precip_prob = 0.35
          elif is_summer:
              temp = np.random.randint(70, 100)
              precip_prob = 0.15
          else:
              temp = np.random.randint(50, 75)
              precip_prob = 0.20

          precipitation = int(np.random.random() < precip_prob)

          if precipitation:
              if is_winter:
                  condition = np.random.choice(['Snowy', 'Rainy', 'Foggy'], p=[0.5, 0.3, 0.2])
              else:
                  condition = np.random.choice(['Rainy', 'Cloudy', 'Foggy'], p=[0.5, 0.3, 0.2])
          else:
              condition = np.random.choice(['Clear', 'Cloudy'], p=[0.7, 0.3])

          if condition in ['Snowy', 'Rainy']:
              wind_speed = np.random.randint(15, 45)
              visibility = np.random.randint(2, 6)
          elif condition == 'Foggy':
              wind_speed = np.random.randint(5, 20)
              visibility = np.random.randint(1, 4)
          else:
              wind_speed = np.random.randint(5, 25)
              visibility = np.random.randint(7, 11)

          return {
              'temperature': temp,
              'humidity': np.random.randint(20, 90),
              'wind_speed': wind_speed,
              'visibility': visibility,
              'precipitation': precipitation,
              'weather_condition': condition
          }

      def generate_dataset(self, num_samples=3000, flight_id_start=20000):
          """
          Generate incremental dataset

          Args:
              num_samples: Number of flight records to generate
              flight_id_start: Starting flight ID (to avoid duplicates with main dataset)

          Returns:
              Pandas DataFrame
          """
          print(f"🔄 Generating {num_samples} incremental flight records...")
          print("   This may take 30-60 seconds...\n")

          data = []

          for i in range(num_samples):
              # Progress indicator
              if (i + 1) % 1000 == 0:
                  print(f"   Generated {i + 1}/{num_samples} records...")

              # Basic flight info
              aircraft_type = np.random.choice(list(self.aircraft_types.keys()))
              aircraft_age = np.random.randint(*self.aircraft_types[aircraft_type]['age_range'])
              base_reliability = self.aircraft_types[aircraft_type]['reliability']

              # Route
              origin = np.random.choice(self.airports)
              destination = np.random.choice([a for a in self.airports if a != origin])
              distance = self.calculate_distance(origin, destination)
              flight_duration = self.estimate_duration(distance)

              # Time factors
              month = np.random.randint(1, 13)
              day_of_week = np.random.randint(0, 7)
              departure_hour = np.random.randint(5, 23)

              # Maintenance
              days_since_maintenance = np.random.randint(1, 180)
              operational_hours = np.random.randint(100, 10000)
              takeoff_landing_cycles = np.random.randint(50, 5000)

              # Sensor data
              sensors = self.generate_sensor_data(aircraft_age, operational_hours, days_since_maintenance)

              # Weather
              origin_weather = self.generate_weather(origin, month)
              dest_weather = self.generate_weather(destination, month)

              # Historical performance
              reliability_factor = base_reliability / 100
              previous_delays = np.random.poisson(5 * (1 - reliability_factor))
              previous_cancellations = np.random.poisson(2 * (1 - reliability_factor))
              previous_maintenance_issues = np.random.poisson(3 * (1 - reliability_factor))
              aircraft_reliability_score = int(np.clip(
                  base_reliability - (aircraft_age * 1.5) + np.random.randint(-5, 5),
                  50, 100
              ))
              crew_experience = np.random.randint(500, 15000)

               # ===== EQUIPMENT FAILURE CALCULATION (REALISTIC 2-5% RATE) =====
              failure_risk = 0.0

              # More realistic thresholds - stricter conditions
              if sensors['engine_temp'] > 320:  # Was 310
                  failure_risk += 0.15  # Was 0.30
              elif sensors['engine_temp'] > 310:  # Was 295
                  failure_risk += 0.08  # Was 0.15

              if sensors['oil_pressure'] < 30:  # Was 35
                  failure_risk += 0.12  # Was 0.25
              elif sensors['oil_pressure'] < 35:  # Was 40
                  failure_risk += 0.06  # Was 0.12

              if sensors['vibration_level'] > 18:  # Was 15
                  failure_risk += 0.12  # Was 0.25
              elif sensors['vibration_level'] > 15:  # Was 12
                  failure_risk += 0.05  # Was 0.10

              if days_since_maintenance > 160:  # Was 150
                  failure_risk += 0.10  # Was 0.20
              elif days_since_maintenance > 140:  # Was 120
                  failure_risk += 0.05  # Was 0.10

              if operational_hours > 9000:  # Was 8000
                  failure_risk += 0.08  # Was 0.15
              elif operational_hours > 7500:  # Was 6000
                  failure_risk += 0.04  # Was 0.08

              if aircraft_age > 20:  # Was 18
                  failure_risk += 0.08  # Was 0.15
              elif aircraft_age > 15:  # Was 12
                  failure_risk += 0.04  # Was 0.07

              if sensors['avionics_health_score'] < 65:  # Was 70
                  failure_risk += 0.06  # Was 0.12

              if takeoff_landing_cycles > 4500:  # Was 4000
                  failure_risk += 0.05  # Was 0.10

              # Cap at lower maximum for realism
              failure_risk = min(failure_risk, 0.50)  # Was 0.85
              equipment_failure = int(np.random.random() < failure_risk)
              # ===== CANCELLATION CALCULATION =====
              cancellation_risk = 0.0

              if equipment_failure:
                  cancellation_risk += 0.50

              if origin_weather['precipitation']:
                  cancellation_risk += 0.15
              if origin_weather['wind_speed'] > 40:
                  cancellation_risk += 0.20
              elif origin_weather['wind_speed'] > 30:
                  cancellation_risk += 0.10
              if origin_weather['visibility'] < 3:
                  cancellation_risk += 0.15

              if dest_weather['precipitation']:
                  cancellation_risk += 0.12
              if dest_weather['wind_speed'] > 40:
                  cancellation_risk += 0.18
              if dest_weather['visibility'] < 3:
                  cancellation_risk += 0.12

              if previous_cancellations > 3:
                  cancellation_risk += 0.12

              if month in [12, 1, 2]:
                  cancellation_risk += 0.08

              if crew_experience < 2000:
                  cancellation_risk += 0.05

              cancellation_risk = min(cancellation_risk, 0.80)
              flight_cancelled = int(np.random.random() < cancellation_risk)

              # ===== CANCELLATION REASON =====
              if flight_cancelled:
                  if equipment_failure:
                      reason = 'Maintenance'
                  elif origin_weather['precipitation'] or dest_weather['precipitation']:
                      reason = 'Weather'
                  elif origin_weather['wind_speed'] > 35 or dest_weather['wind_speed'] > 35:
                      reason = 'Weather'
                  elif crew_experience < 1500:
                      reason = 'Crew'
                  else:
                      reason = np.random.choice(['Operational', 'Weather', 'Other'], p=[0.5, 0.3, 0.2])
              else:
                  reason = 'None'

              # ===== DELAY CALCULATION =====
              if not flight_cancelled:
                  delay_prob = 0.3
                  if origin_weather['wind_speed'] > 25:
                      delay_prob += 0.2
                  if sensors['engine_temp'] > 295:
                      delay_prob += 0.15

                  if np.random.random() < delay_prob:
                      delay_minutes = np.random.choice(
                          [15, 30, 45, 60, 90, 120],
                          p=[0.4, 0.25, 0.15, 0.10, 0.07, 0.03]
                      )
                  else:
                      delay_minutes = 0
              else:
                  delay_minutes = 0

              # ===== COMPILE RECORD =====
              # Use different flight ID range to avoid conflicts with main dataset
              record = {
                  'flight_id': f'FL{flight_id_start + i:05d}',
                  'aircraft_id': f'N{np.random.randint(10000, 99999)}',
                  'airline': np.random.choice(self.airlines),
                  'origin_airport': origin,
                  'destination_airport': destination,
                  'aircraft_type': aircraft_type,
                  'aircraft_age': aircraft_age,
                  **sensors,
                  'operational_hours': operational_hours,
                  'month': month,
                  'day_of_week': day_of_week,
                  'departure_hour': departure_hour,
                  'distance': distance,
                  'flight_duration': flight_duration,
                  'takeoff_landing_cycles': takeoff_landing_cycles,
                  'days_since_last_maintenance': days_since_maintenance,
                  'crew_experience': crew_experience,
                  'origin_temperature': origin_weather['temperature'],
                  'origin_humidity': origin_weather['humidity'],
                  'origin_wind_speed': origin_weather['wind_speed'],
                  'origin_visibility': origin_weather['visibility'],
                  'origin_precipitation': origin_weather['precipitation'],
                  'origin_weather_condition': origin_weather['weather_condition'],
                  'destination_temperature': dest_weather['temperature'],
                  'destination_humidity': dest_weather['humidity'],
                  'destination_wind_speed': dest_weather['wind_speed'],
                  'destination_visibility': dest_weather['visibility'],
                  'destination_precipitation': dest_weather['precipitation'],
                  'destination_weather_condition': dest_weather['weather_condition'],
                  'previous_delays_count': previous_delays,
                  'previous_cancellations_count': previous_cancellations,
                  'previous_maintenance_issues': previous_maintenance_issues,
                  'aircraft_reliability_score': aircraft_reliability_score,
                  'equipment_failure': equipment_failure,
                  'flight_cancelled': flight_cancelled,
                  'cancellation_reason': reason,
                  'delay_minutes': delay_minutes
              }

              data.append(record)

          df = pd.DataFrame(data)
          return df


def generate_incremental_data(num_samples=3000):
      """
      Main function to generate and save incremental dataset with timestamp

      Args:
          num_samples: Number of flight records to generate (default: 3,000)

      USAGE:
          # Generate 3,000 records
          generate_incremental_data(3000)

          # Generate 5,000 records
          generate_incremental_data(5000)

          # Generate 1,000 records
          generate_incremental_data(1000)
      """
      print("\n" + "="*70)
      print("📦 INCREMENTAL DATASET GENERATOR (Multi-File Mode)")
      print("="*70)
      print(f"📊 Generating {num_samples:,} records...\n")

      # Create directory if it doesn't exist
      os.makedirs('backend/data/raw', exist_ok=True)

      # Generate timestamp for filename
      timestamp = datetime.now().strftime("%Y_%m_%d_%H%M%S")

      # Generate data with timestamp-based seed
      generator = IncrementalDataGenerator(use_timestamp_seed=True)
      df = generator.generate_dataset(num_samples=num_samples)

      # Create filename with timestamp
      filename = f'incremental_data_{timestamp}.csv'
      output_path = f'backend/data/raw/{filename}'

      # Save to CSV
      df.to_csv(output_path, index=False)

      # ===== PRINT STATISTICS =====
      print("\n" + "="*70)
      print("📊 INCREMENTAL DATASET STATISTICS")
      print("="*70)
      print(f"🕒 Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
      print(f"📁 Filename: {filename}")
      print(f"📏 Total Records: {len(df):,}")

      print(f"\n🔧 Equipment Failure Rate: {df['equipment_failure'].mean()*100:.2f}%")
      print(f"   - Failures: {df['equipment_failure'].sum():,}")
      print(f"   - Normal: {(~df['equipment_failure'].astype(bool)).sum():,}")

      print(f"\n✈️ Flight Cancellation Rate: {df['flight_cancelled'].mean()*100:.2f}%")
      print(f"   - Cancelled: {df['flight_cancelled'].sum():,}")
      print(f"   - Completed: {(~df['flight_cancelled'].astype(bool)).sum():,}")

      print(f"\n📋 Cancellation Reasons:")
      if df['flight_cancelled'].sum() > 0:
          reason_counts = df[df['flight_cancelled']==1]['cancellation_reason'].value_counts()
          for reason, count in reason_counts.items():
              percentage = count/df['flight_cancelled'].sum()*100
              print(f"   - {reason}: {count} ({percentage:.1f}%)")

      print(f"\n💾 Data saved to: {output_path}")
      print(f"📦 File size: ~{os.path.getsize(output_path) / (1024*1024):.2f} MB")
      print("="*70)
      print("\n✅ INCREMENTAL DATASET GENERATION COMPLETE!")
      print(f"💡 TIP: Run this script again to generate ANOTHER dataset")
      print(f"         Each run creates a NEW file with a different timestamp")
      print("="*70 + "\n")

      return df


  # ===== RUN THE GENERATOR =====
if __name__ == '__main__':
      # ====================================================
      # CUSTOMIZE HERE: Change the number to generate more/less records
      # ====================================================

      generate_incremental_data(num_samples=3000)  # ← Change 3000 to any number you want

      # EXAMPLES:
      # generate_incremental_data(num_samples=1000)   # 1,000 records
      # generate_incremental_data(num_samples=2000)   # 2,000 records
      # generate_incremental_data(num_samples=5000)   # 5,000 records
# python backend/data/generate_incremental_data.py -to run
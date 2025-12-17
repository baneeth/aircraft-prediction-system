import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import os

class AircraftDataGenerator:
    def __init__(self, num_records=15000, random_seed=42):
        """
        Initialize the Aircraft Data Generator

        Args:
            num_records: Number of flight records to generate
            random_seed: Random seed for reproducibility
        """
        self.num_records = num_records
        np.random.seed(random_seed)
        random.seed(random_seed)

        # Define aircraft types and their characteristics
        self.aircraft_types = {
            'Boeing 737': {'age_range': (0, 25), 'capacity': 180},
            'Airbus A320': {'age_range': (0, 20), 'capacity': 160},
            'Boeing 777': {'age_range': (0, 30), 'capacity': 350},
            'Airbus A350': {'age_range': (0, 10), 'capacity': 300},
            'Boeing 787': {'age_range': (0, 12), 'capacity': 280}
        }

        # Airlines and routes
        self.airlines = ['AirGlobal', 'SkyConnect', 'OceanAir', 'Continental Express', 'PacificWings']
        self.routes = [
            ('JFK', 'LAX'), ('ORD', 'SFO'), ('ATL', 'DEN'), ('DFW', 'SEA'),
            ('MIA', 'BOS'), ('LAS', 'PHX'), ('IAH', 'MCO'), ('DCA', 'SAN')
        ]

        # Weather conditions
        self.weather_conditions = ['Clear', 'Cloudy', 'Rainy', 'Stormy', 'Foggy']

    def calculate_distance(self, origin, destination):
        """Calculate approximate flight distance (simplified)"""
        # Simplified distance calculation (in miles)
        distances = {
            ('JFK', 'LAX'): 2475, ('ORD', 'SFO'): 1846, ('ATL', 'DEN'): 1199,
            ('DFW', 'SEA'): 1660, ('MIA', 'BOS'): 1258, ('LAS', 'PHX'): 256,
            ('IAH', 'MCO'): 848, ('DCA', 'SAN'): 2279
        }
        return distances.get((origin, destination), 1500)

    def estimate_duration(self, distance):
        """Estimate flight duration based on distance"""
        avg_speed = 500  # mph
        duration = distance / avg_speed
        return round(duration + np.random.normal(0, 0.3), 2)

    def generate_weather(self):
        """Generate weather conditions with realistic probabilities"""
        return np.random.choice(
            self.weather_conditions,
            p=[0.50, 0.25, 0.15, 0.05, 0.05]  # Clear weather is most common
        )

    def generate_sensor_data(self, aircraft_age, operational_hours, days_since_maintenance):
        """
        Generate sensor readings with ADDITIVE degradation (FIXED VERSION)

        This method creates realistic sensor values that increase additively
        based on aircraft age, operational hours, and maintenance schedule.
        """
        # BASE VALUES (normal operating conditions)
        base_engine_temp = 280  # Celsius
        base_oil_pressure = 45  # PSI
        base_vibration = 10     # mm/s

        # ADDITIVE DEGRADATION FACTORS (FIXED!)
        # Calculate temperature increase
        temp_increase = 0
        temp_increase += (aircraft_age / 100) * 25      # Age contributes up to +25°C
        temp_increase += (days_since_maintenance / 500) * 35  # Maintenance contributes up to +35°C
        temp_increase += (operational_hours / 20000) * 15     # Hours contribute up to +15°C

        # Calculate oil pressure decrease
        pressure_decrease = 0
        pressure_decrease += (aircraft_age / 100) * 8    # Age contributes up to -8 PSI
        pressure_decrease += (days_since_maintenance / 500) * 12  # Maintenance contributes up to -12 PSI
        pressure_decrease += (operational_hours / 20000) * 5      # Hours contribute up to -5 PSI

        # Calculate vibration increase
        vibration_increase = 0
        vibration_increase += (aircraft_age / 100) * 5   # Age contributes up to +5 mm/s
        vibration_increase += (days_since_maintenance / 500) * 8  # Maintenance contributes up to +8 mm/s
        vibration_increase += (operational_hours / 20000) * 3     # Hours contribute up to +3 mm/s

        # Generate final sensor values with realistic noise and hard limits
        engine_temp = np.clip(
            np.random.normal(base_engine_temp + temp_increase, 15),
            240,  # Minimum safe operating temp
            340   # Maximum safe operating temp
        )

        oil_pressure = np.clip(
            np.random.normal(base_oil_pressure - pressure_decrease, 5),
            15,   # Minimum safe pressure
            55    # Maximum safe pressure
        )

        vibration_level = np.clip(
            np.random.normal(base_vibration + vibration_increase, 3),
            5,    # Minimum vibration
            30    # Maximum vibration
        )

        return engine_temp, oil_pressure, vibration_level

    def generate_dataset(self):
        """Generate complete dataset with all features"""
        data = []

        for i in range(self.num_records):
            # Basic flight info
            flight_id = f"FL{10000 + i}"
            aircraft_type = random.choice(list(self.aircraft_types.keys()))
            airline = random.choice(self.airlines)
            origin, destination = random.choice(self.routes)

            # Aircraft characteristics
            age_min, age_max = self.aircraft_types[aircraft_type]['age_range']
            aircraft_age = np.random.randint(age_min, age_max + 1)
            operational_hours = np.random.randint(1000, 50000)

            # Maintenance info
            days_since_maintenance = np.random.randint(1, 500)
            maintenance_count = np.random.randint(5, 50)

            # Flight details
            distance = self.calculate_distance(origin, destination)
            flight_duration = self.estimate_duration(distance)

            # Weather
            weather = self.generate_weather()
            weather_severity = {
                'Clear': 1, 'Cloudy': 2, 'Rainy': 3, 'Stormy': 5, 'Foggy': 4
            }[weather]

            # Sensor data (with FIXED additive degradation)
            engine_temp, oil_pressure, vibration_level = self.generate_sensor_data(
                aircraft_age, operational_hours, days_since_maintenance
            )

            # Additional sensors
            fuel_consumption = np.random.uniform(800, 1500) * (distance / 1000)
            hydraulic_pressure = np.random.uniform(2800, 3200)
            cabin_pressure = np.random.uniform(10.5, 11.5)

            # Operational factors
            previous_delays = np.random.randint(0, 10)
            crew_experience = np.random.uniform(2, 25)  # years

            # TARGET VARIABLES - Equipment Failure (12% rate)
            # Higher risk factors increase failure probability
            failure_risk = 0

            # Age risk (0-30%)
            if aircraft_age > 20:
                failure_risk += 0.30
            elif aircraft_age > 15:
                failure_risk += 0.20
            elif aircraft_age > 10:
                failure_risk += 0.10

            # Maintenance risk (0-25%)
            if days_since_maintenance > 400:
                failure_risk += 0.25
            elif days_since_maintenance > 300:
                failure_risk += 0.15
            elif days_since_maintenance > 200:
                failure_risk += 0.08

            # Sensor risk (0-35%)
            if engine_temp > 320 or oil_pressure < 25 or vibration_level > 25:
                failure_risk += 0.35
            elif engine_temp > 310 or oil_pressure < 30 or vibration_level > 22:
                failure_risk += 0.20
            elif engine_temp > 300 or oil_pressure < 35 or vibration_level > 18:
                failure_risk += 0.10

            # Weather risk (0-10%)
            if weather in ['Stormy', 'Foggy']:
                failure_risk += 0.10
            elif weather == 'Rainy':
                failure_risk += 0.05

            # Base failure rate is 12%, adjusted by risk factors
            base_failure_rate = 0.12
            equipment_failure = 1 if np.random.random() < (base_failure_rate + failure_risk * 0.15) else 0

            # Flight Cancellation (8% rate, highly correlated with equipment failure)
            cancellation_risk = 0.05 if equipment_failure == 0 else 0.50

            # Additional cancellation factors
            if weather == 'Stormy':
                cancellation_risk += 0.15
            elif weather in ['Foggy', 'Rainy']:
                cancellation_risk += 0.08

            if previous_delays > 7:
                cancellation_risk += 0.10

            flight_cancelled = 1 if np.random.random() < cancellation_risk else 0

            # Append record
            data.append({
                'flight_id': flight_id,
                'aircraft_type': aircraft_type,
                'aircraft_age': aircraft_age,
                'operational_hours': operational_hours,
                'days_since_maintenance': days_since_maintenance,
                'maintenance_count': maintenance_count,
                'airline': airline,
                'origin': origin,
                'destination': destination,
                'distance': distance,
                'flight_duration': flight_duration,
                'weather_condition': weather,
                'weather_severity': weather_severity,
                'engine_temperature': round(engine_temp, 2),
                'oil_pressure': round(oil_pressure, 2),
                'vibration_level': round(vibration_level, 2),
                'fuel_consumption': round(fuel_consumption, 2),
                'hydraulic_pressure': round(hydraulic_pressure, 2),
                'cabin_pressure': round(cabin_pressure, 2),
                'previous_delays': previous_delays,
                'crew_experience': round(crew_experience, 2),
                'equipment_failure': equipment_failure,
                'flight_cancelled': flight_cancelled
            })

        return pd.DataFrame(data)


def generate_comprehensive_flight_data(num_records=15000, output_path=None):
    """
    Main function to generate comprehensive aircraft flight data

    Args:
        num_records: Number of flight records to generate
        output_path: Path to save the CSV file

    Returns:
        DataFrame with generated flight data
    """
    if output_path is None:
        output_path = 'backend/data/raw/aircraft_flights_data.csv'

    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Generate data
    generator = AircraftDataGenerator(num_records=num_records)
    df = generator.generate_dataset()

    # Save to CSV
    df.to_csv(output_path, index=False)

    # Print summary statistics
    print(f"\n{'='*60}")
    print(f"AIRCRAFT FLIGHT DATA GENERATION COMPLETE")
    print(f"{'='*60}\n")
    print(f"Total Records: {len(df):,}")
    print(f"Output File: {output_path}\n")

    print("TARGET VARIABLE DISTRIBUTION:")
    print(f"  Equipment Failures: {df['equipment_failure'].sum():,} ({df['equipment_failure'].mean()*100:.1f}%)")
    print(f"  Flight Cancellations: {df['flight_cancelled'].sum():,} ({df['flight_cancelled'].mean()*100:.1f}%)\n")

    print("SENSOR DATA RANGES:")
    print(f"  Engine Temperature: {df['engine_temperature'].min():.1f}°C - {df['engine_temperature'].max():.1f}°C")
    print(f"  Oil Pressure: {df['oil_pressure'].min():.1f} - {df['oil_pressure'].max():.1f} PSI")
    print(f"  Vibration Level: {df['vibration_level'].min():.1f} - {df['vibration_level'].max():.1f} mm/s\n")

    print("AIRCRAFT CHARACTERISTICS:")
    print(f"  Aircraft Age: {df['aircraft_age'].min()} - {df['aircraft_age'].max()} years")
    print(f"  Days Since Maintenance: {df['days_since_maintenance'].min()} - {df['days_since_maintenance'].max()} days\n")

    print(f"{'='*60}\n")

    return df


if __name__ == '__main__':
    # Generate 15,000 records for initial training
    df = generate_comprehensive_flight_data(num_records=15000)
    print("✅ Data generation complete! Ready for preprocessing and model training.")

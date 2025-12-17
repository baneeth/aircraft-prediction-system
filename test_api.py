"""
Simple API Test Script
Tests all endpoints of the Aircraft Prediction API
"""

import requests
import json

API_URL = "http://localhost:5000"

print("\n" + "=" * 70)
print("TESTING AIRCRAFT PREDICTION API")
print("=" * 70)

# TEST 1: Health Check
print("\n[TEST 1] Health Check (GET /)...")
try:
    response = requests.get(f"{API_URL}/")
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
except Exception as e:
    print(f"ERROR: {e}")

# TEST 2: Model Info
print("\n[TEST 2] Model Info (GET /api/models/info)...")
try:
    response = requests.get(f"{API_URL}/api/models/info")
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
except Exception as e:
    print(f"ERROR: {e}")

# TEST 3: Single Prediction
print("\n[TEST 3] Single Prediction (POST /api/predict)...")
try:
    test_flight = {
        "aircraft_type": "Boeing 737",
        "aircraft_age": 15,
        "operational_hours": 25000,
        "days_since_maintenance": 120,
        "maintenance_count": 25,
        "airline": "AirGlobal",
        "origin": "JFK",
        "destination": "LAX",
        "distance": 2475,
        "flight_duration": 5.2,
        "weather_condition": "Clear",
        "weather_severity": 1,
        "engine_temperature": 295.5,
        "oil_pressure": 42.3,
        "vibration_level": 12.8,
        "fuel_consumption": 3500,
        "hydraulic_pressure": 3050,
        "cabin_pressure": 11.2,
        "previous_delays": 2,
        "crew_experience": 12.5,
    }

    response = requests.post(f"{API_URL}/api/predict", json=test_flight)
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
except Exception as e:
    print(f"ERROR: {e}")

# TEST 4: High Risk Flight
print("\n[TEST 4] High Risk Flight (old aircraft, overdue maintenance)...")
try:
    high_risk_flight = {
        "aircraft_type": "Boeing 737",
        "aircraft_age": 25,  # Very old
        "operational_hours": 45000,  # High hours
        "days_since_maintenance": 450,  # Overdue!
        "maintenance_count": 10,
        "airline": "AirGlobal",
        "origin": "JFK",
        "destination": "LAX",
        "distance": 2475,
        "flight_duration": 5.2,
        "weather_condition": "Stormy",  # Bad weather
        "weather_severity": 5,
        "engine_temperature": 330.0,  # High temp!
        "oil_pressure": 25.0,  # Low pressure!
        "vibration_level": 28.0,  # High vibration!
        "fuel_consumption": 3500,
        "hydraulic_pressure": 2900,
        "cabin_pressure": 10.8,
        "previous_delays": 8,
        "crew_experience": 3.0,
    }

    response = requests.post(f"{API_URL}/api/predict", json=high_risk_flight)
    print(f"Status Code: {response.status_code}")
    result = response.json()
    print(
        f"Equipment Failure: {result['predictions']['equipment_failure']['prediction']} "
        f"({result['predictions']['equipment_failure']['probability']}% - "
        f"{result['predictions']['equipment_failure']['risk_level']} Risk)"
    )
    print(
        f"Flight Cancellation: {result['predictions']['flight_cancellation']['prediction']} "
        f"({result['predictions']['flight_cancellation']['probability']}% - "
        f"{result['predictions']['flight_cancellation']['risk_level']} Risk)"
    )
except Exception as e:
    print(f"ERROR: {e}")

print("\n" + "=" * 70)
print("API TESTING COMPLETE!")
print("=" * 70 + "\n")
#python test_api.py -to run

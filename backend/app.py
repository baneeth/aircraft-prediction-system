"""
================================================================================
AIRCRAFT PREDICTION API - BACKEND SERVER
================================================================================

PURPOSE:
--------
Flask REST API that serves predictions from trained ML models.
Allows frontend/users to get equipment failure and flight cancellation predictions.

HOW IT WORKS:
-------------
1. Loads trained models and preprocessing pipeline at startup
2. Receives flight data via HTTP POST requests (JSON format)
3. Preprocesses the data (scales numbers, encodes categories)
4. Runs predictions using XGBoost models
5. Returns prediction results as JSON

ENDPOINTS:
----------
GET  /                    - Health check (is server running?)
GET  /api/models/info     - Get information about loaded models
POST /api/predict         - Single flight prediction
POST /api/batch-predict   - Multiple flights prediction

EXAMPLE REQUEST (POST /api/predict):
------------------------------------
{
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
    "crew_experience": 12.5
}

EXAMPLE RESPONSE:
-----------------
{
    "success": true,
    "predictions": {
        "equipment_failure": {
            "prediction": "No Failure",
            "probability": 15.3,
            "risk_level": "Low"
        },
        "flight_cancellation": {
            "prediction": "On Schedule",
            "probability": 12.8,
            "risk_level": "Low"
        }
    }
}

================================================================================
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd
import numpy as np
import os
from datetime import datetime
import json

app = Flask(__name__)
CORS(app)  # Enable Cross-Origin requests (allows frontend to call API)

# ========== LOAD MODELS AT STARTUP ==========
MODEL_DIR = 'models/saved_models'
PIPELINE_PATH = os.path.join(MODEL_DIR, 'preprocessing_pipeline.pkl')
EQUIPMENT_MODEL_PATH = os.path.join(MODEL_DIR, 'equipment_failure_xgboost_model.pkl')
CANCELLATION_MODEL_PATH = os.path.join(MODEL_DIR, 'flight_cancellation_xgboost_model.pkl')

print("\n" + "="*70)
print("LOADING MODELS...")
print("="*70)

try:
    pipeline = joblib.load(PIPELINE_PATH)
    equipment_model = joblib.load(EQUIPMENT_MODEL_PATH)
    cancellation_model = joblib.load(CANCELLATION_MODEL_PATH)
    print("✓ Preprocessing pipeline loaded")
    print("✓ Equipment failure model loaded")
    print("✓ Flight cancellation model loaded")
    print("="*70 + "\n")
except Exception as e:
    print(f"ERROR loading models: {e}")
    pipeline = None
    equipment_model = None
    cancellation_model = None


# ========== ENDPOINT 1: HEALTH CHECK ==========
@app.route('/', methods=['GET'])
def home():
    """Check if API is running"""
    return jsonify({
        'status': 'online',
        'message': 'Aircraft Prediction API is running',
        'version': '1.0',
        'models_loaded': all([pipeline, equipment_model, cancellation_model]),
        'endpoints': {
            'GET /': 'Health check',
            'GET /api/models/info': 'Model information',
            'POST /api/predict': 'Single flight prediction',
            'POST /api/batch-predict': 'Batch predictions'
        }
    })


# ========== ENDPOINT 2: MODEL INFO ==========
@app.route('/api/models/info', methods=['GET'])
def model_info():
    """Get information about loaded models"""
    if not all([pipeline, equipment_model, cancellation_model]):
        return jsonify({'error': 'Models not loaded'}), 500

    return jsonify({
        'preprocessing_pipeline': 'Loaded',
        'equipment_failure_model': type(equipment_model).__name__,
        'flight_cancellation_model': type(cancellation_model).__name__,
        'required_features': [
            'aircraft_type', 'aircraft_age', 'operational_hours',
            'days_since_maintenance', 'maintenance_count', 'airline',
            'origin', 'destination', 'distance', 'flight_duration',
            'weather_condition', 'weather_severity', 'engine_temperature',
            'oil_pressure', 'vibration_level', 'fuel_consumption',
            'hydraulic_pressure', 'cabin_pressure', 'previous_delays',
            'crew_experience'
        ]
    })


# ========== ENDPOINT 3: SINGLE PREDICTION ==========
@app.route('/api/predict', methods=['POST'])
def predict():
    """Make prediction for a single flight"""
    if not all([pipeline, equipment_model, cancellation_model]):
        return jsonify({'error': 'Models not loaded'}), 500

    try:
        # Get input data
        data = request.get_json()

        # Required fields
        required_fields = [
            'aircraft_type', 'aircraft_age', 'operational_hours',
            'days_since_maintenance', 'maintenance_count', 'airline',
            'origin', 'destination', 'distance', 'flight_duration',
            'weather_condition', 'weather_severity', 'engine_temperature',
            'oil_pressure', 'vibration_level', 'fuel_consumption',
            'hydraulic_pressure', 'cabin_pressure', 'previous_delays',
            'crew_experience'
        ]

        # Check for missing fields
        missing_fields = [field for field in required_fields if field not in data]
        if missing_fields:
            return jsonify({
                'error': 'Missing required fields',
                'missing_fields': missing_fields
            }), 400

        # Convert to DataFrame
        df = pd.DataFrame([data])

        # Preprocess
        X_processed = pipeline.transform(df)

        # Make predictions - Equipment Failure
        equipment_prob = equipment_model.predict_proba(X_processed)[0][1]
        equipment_pred = int(equipment_model.predict(X_processed)[0])

        # Make predictions - Flight Cancellation
        cancellation_prob = cancellation_model.predict_proba(X_processed)[0][1]
        cancellation_pred = int(cancellation_model.predict(X_processed)[0])

        # Calculate risk levels
        def get_risk_level(prob):
            if prob > 0.7:
                return 'High'
            elif prob > 0.4:
                return 'Medium'
            else:
                return 'Low'

        # Return predictions
        return jsonify({
            'success': True,
            'timestamp': datetime.now().isoformat(),
            'predictions': {
                'equipment_failure': {
                    'prediction': 'Failure' if equipment_pred == 1 else 'No Failure',
                    'probability': round(equipment_prob * 100, 2),
                    'risk_level': get_risk_level(equipment_prob)
                },
                'flight_cancellation': {
                    'prediction': 'Cancelled' if cancellation_pred == 1 else 'On Schedule',
                    'probability': round(cancellation_prob * 100, 2),
                    'risk_level': get_risk_level(cancellation_prob)
                }
            },
            'input_summary': {
                'aircraft': f"{data['aircraft_type']} (Age: {data['aircraft_age']} years)",
                'route': f"{data['origin']} → {data['destination']}",
                'maintenance': f"{data['days_since_maintenance']} days ago"
            }
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


# ========== ENDPOINT 4: BATCH PREDICTIONS ==========
@app.route('/api/batch-predict', methods=['POST'])
def batch_predict():
    """Make predictions for multiple flights"""
    if not all([pipeline, equipment_model, cancellation_model]):
        return jsonify({'error': 'Models not loaded'}), 500

    try:
        # Get input data (array of flight objects)
        data = request.get_json()

        if not isinstance(data, list):
            return jsonify({'error': 'Expected array of flight data'}), 400

        # Convert to DataFrame
        df = pd.DataFrame(data)

        # Preprocess
        X_processed = pipeline.transform(df)

        # Make predictions
        equipment_probs = equipment_model.predict_proba(X_processed)[:, 1]
        equipment_preds = equipment_model.predict(X_processed)

        cancellation_probs = cancellation_model.predict_proba(X_processed)[:, 1]
        cancellation_preds = cancellation_model.predict(X_processed)

        # Prepare results
        results = []
        for i in range(len(data)):
            results.append({
                'flight_index': i,
                'equipment_failure': {
                    'prediction': 'Failure' if equipment_preds[i] == 1 else 'No Failure',
                    'probability': round(equipment_probs[i] * 100, 2)
                },
                'flight_cancellation': {
                    'prediction': 'Cancelled' if cancellation_preds[i] == 1 else 'On Schedule',
                    'probability': round(cancellation_probs[i] * 100, 2)
                }
            })

        return jsonify({
            'success': True,
            'timestamp': datetime.now().isoformat(),
            'total_flights': len(data),
            'predictions': results,
            'summary': {
                'total_failures_predicted': int(equipment_preds.sum()),
                'total_cancellations_predicted': int(cancellation_preds.sum())
            }
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500
         # ========== ENDPOINT 5: ANALYTICS ==========
@app.route('/api/analytics', methods=['GET'])
def get_analytics():
      """Get comprehensive analytics data"""
      try:
          analytics_path = 'analytics/analytics_results.json'

          if not os.path.exists(analytics_path):
              return jsonify({
                  'error': 'Analytics data not generated yet. Please run analytics_generator.py first.'
              }), 404

          with open(analytics_path, 'r') as f:
              analytics_data = json.load(f)

          return jsonify({
              'success': True,
              'data': analytics_data
          })

      except Exception as e:
          return jsonify({
              'success': False,
              'error': str(e)
          }), 500



# ========== START SERVER ==========
if __name__ == '__main__':
    print("\n" + "="*70)
    print("AIRCRAFT PREDICTION API SERVER")
    print("="*70)
    print("Server starting...")
    print("API will be available at: http://localhost:5000")
    print("\nEndpoints:")
    print("  GET  /                    - Health check")
    print("  GET  /api/models/info     - Model information")
    print("  POST /api/predict         - Single prediction")
    print("  POST /api/batch-predict   - Batch predictions")
    print("="*70 + "\n")

    app.run(debug=True, host='0.0.0.0', port=5000)
#  python backend/api/app.py - to run

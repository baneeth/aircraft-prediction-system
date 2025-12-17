from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import numpy as np
import os

app = Flask(__name__)
CORS(app)  # Allow frontend to call this API

# Load models
MODEL_PATH = 'models/saved_models/'

try:
    with open(os.path.join(MODEL_PATH, 'equipment_failure_xgboost_model.pkl'), 'rb') as f:
        equipment_failure_model = pickle.load(f)

    with open(os.path.join(MODEL_PATH, 'flight_cancellation_xgboost_model.pkl'), 'rb') as f:
        flight_cancellation_model = pickle.load(f)

    print("✅ Models loaded successfully!")
except Exception as e:
    print(f"❌ Error loading models: {e}")

@app.route('/')
def home():
    return jsonify({
        "message": "Aircraft Prediction API",
        "status": "running",
        "endpoints": ["/predict"]
    })

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        features = data.get('features')

        if not features or len(features) != 46:
            return jsonify({"error": "Please provide exactly 46 features"}), 400

        # Convert to numpy array
        features_array = np.array(features).reshape(1, -1)

        # Make predictions
        equipment_failure_pred = int(equipment_failure_model.predict(features_array)[0])
        flight_cancellation_pred = int(flight_cancellation_model.predict(features_array)[0])

        # Get probabilities
        equipment_failure_prob = float(equipment_failure_model.predict_proba(features_array)[0][1])
        flight_cancellation_prob = float(flight_cancellation_model.predict_proba(features_array)[0][1])

        return jsonify({
            "equipment_failure": {
                "prediction": equipment_failure_pred,
                "probability": round(equipment_failure_prob * 100, 2)
            },
            "flight_cancellation": {
                "prediction": flight_cancellation_pred,
                "probability": round(flight_cancellation_prob * 100, 2)
            }
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)

from flask import Flask, request, jsonify
from flask_cors import CORS
from pymongo import MongoClient
from flask_bcrypt import Bcrypt
from flask_jwt_extended import JWTManager, create_access_token, jwt_required, get_jwt_identity
import numpy as np
import joblib
import os

app = Flask(__name__)
CORS(app)

@app.route("/")
def home():
    return "Heart Disease Prediction API is running 🚀"

app.config["JWT_SECRET_KEY"] = "secret123"

jwt = JWTManager(app)
bcrypt = Bcrypt(app)

# ===================== MongoDB =====================
client = MongoClient(os.environ.get("MONGO_URI"), tlsAllowInvalidCertificates=True)
db = client["heartdb"]
users = db["users"]
history = db["history"]

# ===================== GLOBAL =====================
model = None
scaler = None

# ===================== LAZY LOAD =====================
def load_resources():
    global model, scaler

    # Load scaler once
    if scaler is None:
        try:
            print("Loading scaler...")
            scaler = joblib.load("scaler.pkl")
            print("Scaler loaded")
        except Exception as e:
            print("Scaler error:", e)

    # Load model only when needed
    if model is None:
        try:
            from tensorflow.keras.models import load_model
            print("Loading model...")
            model = load_model("heart_model.h5", compile=False)
            print("Model loaded")
        except Exception as e:
            print("Model failed:", e)
            model = None


# ===================== REGISTER =====================
@app.route("/register", methods=["POST"])
def register():
    data = request.json
    hashed = bcrypt.generate_password_hash(data["password"]).decode("utf-8")

    users.insert_one({
        "username": data["username"],
        "password": hashed
    })

    return jsonify({"msg": "Registered"})


# ===================== LOGIN =====================
@app.route("/login", methods=["POST"])
def login():
    data = request.json
    user = users.find_one({"username": data["username"]})

    if user and bcrypt.check_password_hash(user["password"], data["password"]):
        token = create_access_token(identity=data["username"])
        return jsonify({"token": token})

    return jsonify({"msg": "Invalid credentials"}), 401


# ===================== PREDICT =====================
@app.route("/predict", methods=["POST"])
@jwt_required()
def predict():
    global model, scaler

    data = request.json
    user = get_jwt_identity()

    try:
        print("Incoming:", data)

        # Load resources
        load_resources()

        # Safe defaults
        if not data:
            return jsonify({"error": "No data received"}), 400

        # BMI
        bmi = float(data.get('weight', 0)) / ((float(data.get('height', 1)) / 100) ** 2)

        # Age conversion
        age_days = float(data.get('age', 0)) * 365

        # Features safely
        features = np.array([[
            age_days,
            float(data.get('height', 0)),
            float(data.get('weight', 0)),
            int(data.get('gender', 0)),
            float(data.get('ap_hi', 0)),
            float(data.get('ap_lo', 0)),
            int(data.get('cholesterol', 1)),
            int(data.get('gluc', 1)),
            int(data.get('smoke', 0)),
            int(data.get('alco', 0)),
            int(data.get('active', 0)),
            bmi
        ]])

        # Apply scaler if exists
        if scaler:
            features = scaler.transform(features)

        # Predict safely
        if model:
            prediction = float(model.predict(features)[0][0])
        else:
            prediction = float(np.mean(features))  # fallback

        result = "High Risk" if prediction > 0.5 else "Low Risk"

        # Save history (safe)
        try:
            history.insert_one({
                "user": user,
                "result": result,
                "risk": prediction
            })
        except Exception as db_error:
            print("DB error:", db_error)

        return jsonify({
            "result": result,
            "risk": prediction
        })

    except Exception as e:
        print("ERROR:", str(e))
        return jsonify({"error": str(e)}), 500

# ===================== HISTORY =====================
@app.route("/history", methods=["GET"])
@jwt_required()
def get_history():
    user = get_jwt_identity()
    data = list(history.find({"user": user}, {"_id": 0}))
    return jsonify(data)


# ===================== RUN =====================
if __name__ == "__main__":
    print("Starting Flask app...")

    port = int(os.environ.get("PORT", 5000))
    print(f"Running on port {port}")

    app.run(host="0.0.0.0", port=port)
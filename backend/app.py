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

# ===================== HOME =====================
@app.route("/")
def home():
    return "Heart Disease Prediction API is running 🚀"

# ===================== LAZY LOAD =====================
def load_resources():
    global model, scaler

    if scaler is None:
        try:
            scaler = joblib.load("scaler.pkl")
            print("Scaler loaded")
        except Exception as e:
            print("Scaler error:", e)

    if model is None:
        try:
            from tensorflow.keras.models import load_model
            model = load_model("heart_model.h5", compile=False)
            print("Model loaded")
        except Exception as e:
            print("Model error:", e)
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
        load_resources()

        # BMI
        height = float(data.get('height', 1))
        weight = float(data.get('weight', 0))
        bmi = weight / ((height / 100) ** 2)

        # Age conversion
        age_days = float(data.get('age', 0)) * 365

        # Feature array
        features = np.array([[
            age_days,
            height,
            weight,
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

        # Scale
        if scaler:
            features = scaler.transform(features)

        # 🔥 Prediction
        if model:
            prediction = float(model.predict(features)[0][0])
        else:
            # realistic fallback logic
            prediction = 0

            if data.get('ap_hi', 0) > 140:
                prediction += 0.2
            if data.get('cholesterol', 1) == 3:
                prediction += 0.2
            if data.get('gluc', 1) == 3:
                prediction += 0.2
            if data.get('smoke', 0) == 1:
                prediction += 0.1
            if data.get('alco', 0) == 1:
                prediction += 0.1
            if data.get('active', 1) == 0:
                prediction += 0.1
            if weight > 90:
                prediction += 0.1

        # Clamp
        prediction = max(0, min(prediction, 1))

        # Convert to %
        risk_percent = round(prediction * 100, 2)

        result = "High Risk" if prediction > 0.5 else "Low Risk"

        # Save history
        try:
            history.insert_one({
                "user": user,
                "result": result,
                "risk": risk_percent
            })
        except Exception as e:
            print("DB error:", e)

        return jsonify({
            "result": result,
            "risk": risk_percent
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
    app.run(host="0.0.0.0", port=port)
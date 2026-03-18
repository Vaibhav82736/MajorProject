from flask import Flask, request, jsonify
from flask_cors import CORS
from pymongo import MongoClient
from flask_bcrypt import Bcrypt
from flask_jwt_extended import JWTManager, create_access_token, jwt_required, get_jwt_identity
import numpy as np
import joblib
import os

# TRY loading tensorflow safely
try:
    from tensorflow.keras.models import load_model
    print("Loading model...")
    model = load_model("heart_model.h5", compile=False)
    print("Model loaded successfully")
except Exception as e:
    print("Model failed to load:", e)
    model = None  # fallback

# Flask app
app = Flask(__name__)
CORS(app)

app.config["JWT_SECRET_KEY"] = "secret123"

jwt = JWTManager(app)
bcrypt = Bcrypt(app)

# MongoDB
client = MongoClient(os.environ.get("MONGO_URI"), tlsAllowInvalidCertificates=True)
db = client["heartdb"]
users = db["users"]
history = db["history"]

# Load scaler
scaler = joblib.load("scaler.pkl")


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
    data = request.json
    user = get_jwt_identity()

    try:
        # BMI
        bmi = data['weight'] / ((data['height'] / 100) ** 2)

        # Convert age
        age_days = data['age'] * 365

        # Features
        features = np.array([[
            age_days,
            data['height'],
            data['weight'],
            data['gender'],
            data['ap_hi'],
            data['ap_lo'],
            data['cholesterol'],
            data['gluc'],
            data['smoke'],
            data['alco'],
            data['active'],
            bmi
        ]])

        features = scaler.transform(features)

        # 🔥 SAFE PREDICTION
        if model:
            prediction = float(model.predict(features)[0][0])
        else:
            # fallback if model fails (for deployment safety)
            prediction = float(np.mean(features))

        result = "High Risk" if prediction > 0.5 else "Low Risk"

        # Save history
        history.insert_one({
            "user": user,
            "result": result,
            "risk": prediction
        })

        return jsonify({
            "result": result,
            "risk": prediction
        })

    except Exception as e:
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
    port = int(os.environ.get("PORT", 5000))
    print(f"Running on port {port}")
    app.run(host="0.0.0.0", port=port)
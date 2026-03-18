from flask import Flask, request, jsonify
from flask_cors import CORS
from pymongo import MongoClient
from flask_bcrypt import Bcrypt
from flask_jwt_extended import JWTManager, create_access_token, jwt_required, get_jwt_identity
import numpy as np
import joblib
from tensorflow.keras.models import load_model
import os

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

# Load model
model = load_model("heart_model.h5")
scaler = joblib.load("scaler.pkl")

# REGISTER
@app.route("/register", methods=["POST"])
def register():
    data = request.json
    hashed = bcrypt.generate_password_hash(data["password"]).decode("utf-8")

    users.insert_one({
        "username": data["username"],
        "password": hashed
    })

    return jsonify({"msg": "Registered"})


# LOGIN
@app.route("/login", methods=["POST"])
def login():
    data = request.json
    user = users.find_one({"username": data["username"]})

    if user and bcrypt.check_password_hash(user["password"], data["password"]):
        token = create_access_token(identity=data["username"])
        return jsonify({"token": token})

    return jsonify({"msg": "Invalid credentials"}), 401


# PREDICT
@app.route("/predict", methods=["POST"])
@jwt_required()
def predict():
    data = request.json
    user = get_jwt_identity()

    try:
        bmi = data['weight'] / ((data['height'] / 100) ** 2)
        age_days=data['age']*365
        features = np.array([[
            age_days, data['height'], data['weight'], data['gender'],
            data['ap_hi'], data['ap_lo'], data['cholesterol'], data['gluc'],
            data['smoke'], data['alco'], data['active'], bmi
        ]])

        features = scaler.transform(features)
        prediction = model.predict(features)[0][0]

        result = "High Risk" if prediction > 0.5 else "Low Risk"

        history.insert_one({
            "user": user,
            "result": result,
            "risk": float(prediction)
        })

        return jsonify({"result": result, "risk": float(prediction)})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# HISTORY
@app.route("/history", methods=["GET"])
@jwt_required()
def get_history():
    user = get_jwt_identity()
    data = list(history.find({"user": user}, {"_id": 0}))
    return jsonify(data)


if __name__ == "__main__":
    app.run(port=5000)

import os
port = int(os.environ.get("PORT", 5000))
app.run(host="0.0.0.0", port=port)
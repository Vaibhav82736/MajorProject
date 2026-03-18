import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

# =========================
# 1. LOAD DATASET
# =========================
df = pd.read_csv("cardio_train.csv", sep=';')

# =========================
# 2. PREPROCESSING
# =========================

# Convert age from days to years
df['age'] = df['age'] / 365

# Remove unrealistic values
df = df[(df['ap_hi'] > 0) & (df['ap_lo'] > 0)]
df = df[df['ap_hi'] < 250]
df = df[df['ap_lo'] < 200]

# Add BMI (important feature)
df['bmi'] = df['weight'] / ((df['height'] / 100) ** 2)

# Features & target
X = df.drop(['cardio', 'id'], axis=1)
y = df['cardio']

# =========================
# 3. SPLIT + SCALE
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# =========================
# 4. BUILD MODEL (MLP)
# =========================
model = Sequential()

model.add(Dense(32, activation='relu', input_shape=(X_train.shape[1],)))
model.add(Dropout(0.3))

model.add(Dense(16, activation='relu'))
model.add(Dropout(0.2))

model.add(Dense(1, activation='sigmoid'))

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# =========================
# 5. TRAIN MODEL
# =========================
model.fit(
    X_train, y_train,
    epochs=40,
    batch_size=32,
    validation_split=0.1
)

# =========================
# 6. EVALUATE
# =========================
loss, accuracy = model.evaluate(X_test, y_test)
print("Test Accuracy:", accuracy)

# =========================
# 7. SAVE MODEL + SCALER
# =========================
model.save("heart_model.h5")
joblib.dump(scaler, "scaler.pkl")

print("✅ Model and scaler saved successfully!")
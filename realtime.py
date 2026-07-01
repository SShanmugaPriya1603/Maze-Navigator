import serial
import numpy as np
import joblib
from tensorflow.keras.models import load_model
import time

PORT = "COM3"
BAUD = 115200
SAMPLES = 50

# Load model and encoder
model = load_model("gesture_model.h5")
le = joblib.load("label_encoder.pkl")

ser = serial.Serial(PORT, BAUD, timeout=2)
time.sleep(2)
print("Ready! Move your hand to detect gestures...")

while True:
    try:
        readings = []
        
        # Collect 1 second of data
        start = time.time()
        while time.time() - start < 1.5:
            try:
                line = ser.readline().decode('utf-8', errors='ignore').strip()
                if line and ',' in line:
                    parts = line.split(",")
                    if len(parts) == 3:
                        x, y, z = map(float, parts)
                        readings.append([x, y, z])
            except:
                continue
        
        if len(readings) >= 50:
            # Prepare input
            data = np.array(readings[:50])
            data = data.reshape(1, 50, 3)
            
            # Predict
            pred = model.predict(data, verbose=0)
            confidence = np.max(pred)
            gesture = le.inverse_transform([np.argmax(pred)])[0]
            
            if confidence > 0.85:
                print(f"Gesture: {gesture.upper()} ({confidence*100:.1f}%)")
                ser.write((gesture.upper() + '\n').encode())
            else:
                print("No clear gesture detected")
        else:
            print(f"Not enough data: {len(readings)} samples")

    except KeyboardInterrupt:
        print("\nStopped.")
        ser.close()
        break
    except Exception as e:
        print(f"Error: {e}")
        continue
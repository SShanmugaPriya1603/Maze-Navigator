import serial
import pandas as pd
import time

PORT = "COM3"        
BAUD = 115200
DURATION = 1         
SAMPLES_PER_SEC = 50 

gestures = ["shake", "tilt_left", "tilt_right", "flick_up"]
data = []

ser = serial.Serial(PORT, BAUD, timeout=2)
time.sleep(2)

for gesture in gestures:
    print(f"\n--- Gesture: {gesture.upper()} ---")
    
    for i in range(50):  # 50 samples per gesture
        input(f"  Sample {i+1}/50 → Press ENTER, then do the gesture")
        
        readings = []
        start = time.time()
        
        while time.time() - start < DURATION:
            line = ser.readline().decode().strip()
            if line:
                try:
                    x, y, z = map(float, line.split(","))
                    readings.append([x, y, z])
                except:
                    pass
        
        if readings:
            # Flatten the 1-second window into one row
            flat = [val for r in readings[:SAMPLES_PER_SEC] for val in r]
            flat.append(gesture)
            data.append(flat)
            print(f"  Captured {len(readings)} readings ✓")

ser.close()

# Save to CSV
columns = [f"{axis}{i}" for i in range(SAMPLES_PER_SEC) for axis in ["x","y","z"]] + ["label"]
df = pd.DataFrame(data, columns=columns)
df.to_csv("gesture_data.csv", index=False)
print("\nDataset saved as gesture_data.csv")
print(df["label"].value_counts())
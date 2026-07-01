"""
gesture_ws_bridge.py
────────────────────────────────────────────────────────────────
Reads MPU6050 data from ESP32 over Serial
  → runs your trained Keras CNN
  → broadcasts gesture via WebSocket to the browser game

Usage:
  python gesture_ws_bridge.py           (auto-detects COM port)
  python gesture_ws_bridge.py --port COM3
  python gesture_ws_bridge.py --demo    (keyboard mode, no hardware)
"""

import asyncio
import json
import time
import threading
import argparse
import sys
import numpy as np

# ── check dependencies ────────────────────────────────────────
try:
    import websockets
except ImportError:
    sys.exit("Missing: pip install websockets")

try:
    import serial
    import serial.tools.list_ports
except ImportError:
    sys.exit("Missing: pip install pyserial")

try:
    from tensorflow.keras.models import load_model
    import joblib
except ImportError:
    sys.exit("Missing: pip install tensorflow joblib")

# ── load your trained model ───────────────────────────────────
print("Loading gesture model...")
try:
    model = load_model("gesture_model.h5")
    le    = joblib.load("label_encoder.pkl")
    print(f"✅ Model loaded | Classes: {list(le.classes_)}")
except FileNotFoundError:
    sys.exit("❌ gesture_model.h5 not found. Run train_model.py first!")

# ── shared state ──────────────────────────────────────────────
clients = set()
current = {"gesture": "idle", "confidence": 0.0, "ts": 0}

async def broadcast(gesture: str, confidence: float):
    current["gesture"]    = gesture
    current["confidence"] = round(float(confidence), 3)
    current["ts"]         = int(time.time() * 1000)
    if clients:
        msg = json.dumps(current)
        await asyncio.gather(*[c.send(msg) for c in list(clients)],
                             return_exceptions=True)

# ── WebSocket server ──────────────────────────────────────────
async def ws_handler(ws):
    clients.add(ws)
    print(f"🎮 Browser connected  (total: {len(clients)})")
    try:
        await ws.send(json.dumps(current))   # send current state immediately
        async for _ in ws:
            pass
    except Exception:
        pass
    finally:
        clients.discard(ws)
        print(f"🎮 Browser disconnected (total: {len(clients)})")

# ── auto-detect ESP32 serial port ────────────────────────────
def find_port():
    for p in serial.tools.list_ports.comports():
        d = (p.description or "").lower()
        if any(k in d for k in ["cp210", "ch340", "ftdi", "uart", "esp", "usb serial"]):
            return p.device
    return None

# ── serial reader + CNN inference ────────────────────────────
async def run_serial(port: str, baud: int, loop):
    print(f"🔌 Opening {port} @ {baud}...")
    try:
        ser = serial.Serial(port, baud, timeout=2)
    except Exception as e:
        sys.exit(f"❌ Cannot open serial port: {e}")

    time.sleep(2)
    print("✅ Serial open — reading gestures...")

    SAMPLES = 50
    readings = []
    last_gesture = "idle"

    while True:
        try:
            raw = await loop.run_in_executor(None, ser.readline)
            line = raw.decode("utf-8", errors="ignore").strip()
            if not line or "," not in line:
                continue
            parts = line.split(",")
            if len(parts) != 3:
                continue
            x, y, z = float(parts[0]), float(parts[1]), float(parts[2])
            readings.append([x, y, z])

            if len(readings) >= SAMPLES:
                data = np.array(readings[:SAMPLES]).reshape(1, SAMPLES, 3)
                pred = model.predict(data, verbose=0)
                conf = float(np.max(pred))
                gesture = le.inverse_transform([np.argmax(pred)])[0]

                readings = readings[10:]   # slide window by 10 (overlap)

                if conf > 0.82:
                    label = gesture.lower()
                    if label != last_gesture or conf > 0.95:
                        print(f"  ✋ {gesture.upper():12s}  {conf*100:.1f}%")
                        asyncio.run_coroutine_threadsafe(
                            broadcast(label, conf), loop)
                        last_gesture = label
                else:
                    if last_gesture != "idle":
                        asyncio.run_coroutine_threadsafe(
                            broadcast("idle", 1.0), loop)
                        last_gesture = "idle"

        except ValueError:
            continue
        except Exception as e:
            print(f"⚠️  {e}")
            await asyncio.sleep(0.01)

# ── keyboard demo mode (no hardware) ─────────────────────────
KEY_MAP = {"a": "tilt_left", "d": "tilt_right",
           "w": "flick_up",  "s": "shake", " ": "idle"}

async def run_demo(loop):
    print("\n🎮 DEMO MODE — no hardware needed!")
    print("   A = Tilt Left   D = Tilt Right")
    print("   W = Flick Up    S = Shake")
    print("   SPACE = Idle    CTRL+C = quit\n")

    def _kb():
        while True:
            ch = sys.stdin.read(1).lower()
            g  = KEY_MAP.get(ch, "idle")
            asyncio.run_coroutine_threadsafe(broadcast(g, 0.99), loop)
            if g != "idle":
                time.sleep(0.4)
                asyncio.run_coroutine_threadsafe(broadcast("idle", 1.0), loop)

    threading.Thread(target=_kb, daemon=True).start()
    while True:
        await asyncio.sleep(1)

# ── main ─────────────────────────────────────────────────────
async def main(args):
    loop = asyncio.get_event_loop()
    print(f"\n🚀 WebSocket server → ws://localhost:{args.ws}")
    print("   Open game.html in your browser, then move your hand!\n")

    async with websockets.serve(ws_handler, "0.0.0.0", args.ws):
        if args.demo:
            await run_demo(loop)
        else:
            port = args.port or find_port()
            if not port:
                print("⚠️  ESP32 not detected. Using demo mode instead.")
                print("   (Specify --port COM3  or use --demo flag)\n")
                await run_demo(loop)
            else:
                await run_serial(port, args.baud, loop)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port",  default=None, help="Serial port e.g. COM3 or /dev/ttyUSB0")
    ap.add_argument("--baud",  type=int, default=115200)
    ap.add_argument("--ws",    type=int, default=8765, help="WebSocket port")
    ap.add_argument("--demo",  action="store_true", help="Keyboard demo, no ESP32 needed")
    args = ap.parse_args()

    # raw terminal for single-key input on Linux/Mac
    if args.demo and sys.platform != "win32":
        import tty, termios
        fd  = sys.stdin.fileno()
        old = termios.tcgetattr(fd)
        tty.setcbreak(fd)
        try:
            asyncio.run(main(args))
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old)
    else:
        try:
            asyncio.run(main(args))
        except KeyboardInterrupt:
            print("\n👋 Stopped.")
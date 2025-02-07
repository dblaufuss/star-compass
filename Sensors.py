import websocket
import json

def get_sensor(target: str, host: str = "ws://localhost:8080") -> dict:
    ws = websocket.WebSocket()
    ws.connect(f"{host}{target}")
    try:
        return json.loads(ws.recv())
    finally:
        ws.close()

def get_orientation() -> dict:
    return get_sensor("/sensor/connect?type=android.sensor.orientation")

def get_location() -> dict:
    return get_sensor("/gps")

if __name__ == "__main__":
    while True:
        print(get_orientation())
        print(get_location())
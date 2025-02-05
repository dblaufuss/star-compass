import websocket
import json

def get_orientation(host) -> dict:
    ws = websocket.WebSocket()
    ws.connect(f"{host}/sensor/connect?type=android.sensor.orientation")
    try:
        return json.loads(ws.recv())
    finally:
        ws.close()

def get_location(host):
    ws = websocket.WebSocket()
    ws.connect(f"{host}/gps")
    try:
        return json.loads(ws.recv())
    finally:
        ws.close()

if __name__ == "__main__":
    server = "ws://localhost:8080"
    print(get_orientation(server))
    print(get_location(server))
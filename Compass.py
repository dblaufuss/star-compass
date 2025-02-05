import websocket

# This must be running on phone with USB debugging enabled
# https://github.com/umer0586/SensorServer

def get_orientation(url) -> dict:
    ws = websocket.WebSocket()
    ws.connect(url)
    try:
        return ws.recv()
    finally:
        ws.close()

server = "ws://localhost:8080/sensor/connect?type=android.sensor.orientation"
print(get_orientation(server))
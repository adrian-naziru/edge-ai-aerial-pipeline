import socket
import struct
import cv2
import numpy as np
from PIL import ImageGrab
import pygetwindow as gw
import time

IMX_IP = "10.39.82.119"

PORT_OUT = 9002     # IMX -> Windows
PORT_IN = 9001      # Windows -> IMX

def find_window():
    wins = gw.getAllWindows()
    for w in wins:
        title = w.title.lower()
        if ("google" in title and "maps" in title) or \
           ("google earth" in title):
            if w.width > 100 and w.height > 100:
                return w
    return None


print("Searching for Google Maps window ")

win = None
while win is None:
    win = find_window()
    if win is None:
        print("Window not found. Open Google Maps and make it visible ")
        time.sleep(1)

print(f" Found window: {win.title}")
x, y, w, h = win.left, win.top, win.width, win.height
print(f" Capturing region: x={x}, y={y}, w={w}, h={h}")

#  CONNECT TO IMX

sock_send = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock_send.connect((IMX_IP, PORT_IN))
print("[OK] Connected TO IMX on port 9001")

#  RECEIVE FROM IMX (RESULT)

sock_recv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock_recv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
sock_recv.bind(("0.0.0.0", PORT_OUT))
sock_recv.listen(1)

print("[OK] Waiting for IMX connection back on port 9002...")
conn_recv, addr = sock_recv.accept()
print("[OK] IMX connected BACK from:", addr)

print("[INFO] Starting capture → send → receive loop... press Q to exit.")

while True:

    x, y, w, h = win.left, win.top, win.width, win.height

    img = ImageGrab.grab(bbox=(x, y, x + w, y + h))

    frame = cv2.cvtColor(np.array(img), cv2.COLOR_BGR2RGB)

    _, encoded = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    data = encoded.tobytes()

    sock_send.sendall(struct.pack(">I", len(data)) + data)

    # segmented result
    size_data = conn_recv.recv(4)
    if not size_data:
        break
    size = struct.unpack(">I", size_data)[0]

    buffer = b""
    while len(buffer) < size:
        recv_chunk = conn_recv.recv(size - len(buffer))
        if not recv_chunk:
            break
        buffer += recv_chunk

    result = cv2.imdecode(np.frombuffer(buffer, dtype=np.uint8), cv2.IMREAD_COLOR)

    # Display
    cv2.imshow("IMX Segmentation Output", result)

    if cv2.waitKey(1) == ord('q'):
        break


cv2.destroyAllWindows()
sock_send.close()
conn_recv.close()
sock_recv.close()

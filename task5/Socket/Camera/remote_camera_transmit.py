import pickle
import socket
import struct
import cv2


class RemoteCamera():
    """Class for transmit frames to clients based on sockets"""

    HOST_IP = ""

    def __init__(self,source: int, port: int = 10100):
        self._camera = cv2.VideoCapture(source)
         # Check is everything fine with video capture object
        if not self._camera.isOpened():
            print("Bad source. Try another camera source index.")
            raise SystemExit

        server_socket = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
        server_socket.bind((self.HOST_IP, port))
        server_socket.listen(5)

        print("Waiting for connection...")
        try:
            self._client_socket, _ = server_socket.accept()
        except KeyboardInterrupt:
            raise SystemExit("Waiting client(s) stopped manualy.")
        print("Connected.")

    def __del__(self):
        try:
            self._client_socket.close()
        except AttributeError:
            pass
        self._camera.release()

    def send_frame(self):
    # Get frame from video/camera
        ret, frame = self._camera.read()
        if not ret:
            print("Camera stopped!")
            raise SystemExit

        try:
            # сериализация
            byte_frame = pickle.dumps(frame)
            # получаем байтовое представления длины frame,
            # и добавляем к нему фрейм
            # b'len_frame'+ b'byte_frame'
            message = struct.pack('Q', len(byte_frame)) + byte_frame
            # отправляем пакет
            self._client_socket.sendall(message)
        except ConnectionResetError:
            print("Connection lost. Interrupt by client.")
            raise SystemExit
        except BrokenPipeError:
            raise SystemExit

try:
    cam = RemoteCamera(0)
except KeyboardInterrupt:
    print("Stopped manualy.")
    exit()

try:
    while True:
        cam.send_frame()
except KeyboardInterrupt:
    pass
except SystemExit:
    pass

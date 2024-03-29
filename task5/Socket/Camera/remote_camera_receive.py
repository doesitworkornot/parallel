import pickle
import socket
import struct

import cv2
import numpy as np


class RemoteViewer:
    """Class for receiving frames from host based on sockets"""

    # размер структуры для длины frame
    PAYLOAD_SIZE = struct.calcsize("Q")

    def __init__(self, ip_address: str, port: int = 10100):
        self._video_client = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
        self._data = bytes()

        print("Connecting...")
        try:
            self._video_client.connect((ip_address, port))
        except ConnectionRefusedError:
            raise SystemExit("Wrong port or the host isn't transmitting frames.")
        except OSError:
            raise SystemExit("There isn't host(s) on this ip address.")
        print("Connected.")

    def __del__(self):
        self._video_client.close()
        cv2.destroyWindow("Receiving frame")

    def get_frame(self) -> np.ndarray:
        """Getting frame from host"""

        try:
            # пока не получим данные о длине фрейма
            while len(self._data) < self.PAYLOAD_SIZE:
                packet = self._video_client.recv(4*1024) # 4K
                if not packet:
                    break
                self._data += packet
        except ConnectionResetError:
            raise SystemExit("Lost connection.")

        # берем из сообщения байтовое представления длины фрейма
        packed_msg_size = self._data[:self.PAYLOAD_SIZE]
        # удалем данные о длине из сообщения
        self._data = self._data[self.PAYLOAD_SIZE:]
        # получаем длину сообщения в виде числа int
        msg_size = struct.unpack("Q",packed_msg_size)[0]

        try:
            # Читаем данные, пока не получим хотя бы один фрейм
            while len(self._data) < msg_size:
                self._data += self._video_client.recv(128*1024)
        except ConnectionResetError:
            raise SystemExit("Lost connection.")

        # забираем фрейм из сообщения
        frame_data = self._data[:msg_size]
        # удаляем фрейм из сообщения
        self._data = self._data[msg_size:]
        # десериализация
        frame = pickle.loads(frame_data)

        return frame

    def show(self, frame):
        """Show getted frame in opencv window"""

        cv2.imshow("Receiving frame", frame)
        cv2.waitKey(1)

    @property
    def host_runnig(self):
        """Checking is host still alive"""

        try:
            # Attempt to send a small piece of data
            self._video_client.sendall(b'ping')
            return True
        except (socket.timeout, BrokenPipeError, ConnectionResetError):
            return False


def main():
    """Receiving frames from host and showing them"""

    image_viewer = RemoteViewer("0.0.0.0")

    print("Start receiving frames.")
    while image_viewer.host_runnig:
        frame = image_viewer.get_frame()
        image_viewer.show(frame)


if __name__ == "__main__":
    main()

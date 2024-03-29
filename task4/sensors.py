import argparse
import multiprocessing
import time

import cv2
import numpy as np


class Sensor():
    def __init__(self, delay):
        self.delay = delay
        self.data = 0

    def get(self):
        time.sleep(self.delay)
        self.data += 1
        return self.data


class cam():
    def __init__(self, resolution, which):
        parts = resolution.split('*')
        self.h = int(parts[0])
        self.w = int(parts[1])
        self.cam = cv2.VideoCapture(which)

    def get(self):
        ret, frame = self.cam.read()
        # frame = cv2.imread("hubble_cgcg396-2_potw2227a.jpg")
        return frame

    def __del__(self):
        self.cam.release()


def sens_process(que, sensor):
    while (1):
        new_val = sensor.get()
        while not que.empty():
            que.get()
        que.put(new_val)


def cam_process(que, cam):
    h = cam.h
    w = cam.w
    while (1):
        new_val = cam.get()
        while not que.empty():
            que.get()
        new_val = cv2.resize(new_val, (h, w), interpolation=cv2.INTER_AREA)
        que.put(new_val)


def parce(s1, s2, s3, sc):
    x = sc.shape[1] - 700  # Правая граница минус смещение по горизонтали
    y = sc.shape[0] - 50  # Нижняя граница минус смещение по вертикали

    font = cv2.FONT_HERSHEY_SIMPLEX
    size_font = 0.9
    thic_font = 2
    color_font = (255, 255, 255)

    text = f"Sensor 1: {s1} Sensor 2: {s2} Sensor 3: {s3}"

    cv2.putText(sc, text, (x, y), font, size_font, color_font, thic_font)

    return sc


def main(args):
    sensor_1 = Sensor(delay=1)
    sensor_2 = Sensor(delay=0.1)
    sensor_3 = Sensor(delay=0.01)
    cam = cam(args.resolution, args.cam)

    que_1 = multiprocessing.Queue()
    que_2 = multiprocessing.Queue()
    que_3 = multiprocessing.Queue()
    que_cam = multiprocessing.Queue()

    proc_1 = multiprocessing.Process(target=sens_process, args=(que_1, sensor_1))
    proc_2 = multiprocessing.Process(target=sens_process, args=(que_2, sensor_2))
    proc_3 = multiprocessing.Process(target=sens_process, args=(que_3, sensor_3))
    proc_cam = multiprocessing.Process(target=cam_process, args=(que_cam, cam))

    proc_1.start()
    proc_2.start()
    proc_3.start()
    proc_cam.start()

    h = cam.h
    w = cam.w
    old_im = np.zeros((w, h, 3), np.uint8)

    old_1 = 0
    old_2 = 0
    old_3 = 0

    while (1):
        if not que_1.empty():
            old_1 = que_1.get()

        if not que_2.empty():
            old_2 = que_2.get()

        if not que_3.empty():
            old_3 = que_3.get()

        if not que_cam.empty():
            old_im = que_cam.get()

        img = parce(old_1, old_2, old_3, old_im)
        cv2.imshow("img", img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

    proc_1.terminate()
    proc_2.terminate()
    proc_3.terminate()
    proc_cam.terminate()

    proc_1.join()
    proc_2.join()
    proc_3.join()
    proc_cam.join()

    del sensor_1
    del sensor_2
    del sensor_3
    del cam



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cam', type=int, default=0, help='Cam Name')
    parser.add_argument('--resolution', type=str, default='900*900', help='Cam res')
    parser.add_argument('--freq', type=int, default=40, help='frq')
    args = parser.parse_args()
    main(args)
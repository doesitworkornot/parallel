import argparse
import multiprocessing
import time

import cv2
import numpy as np


class Сенсор():
    def __init__(self, делэй):
        self.делэй = делэй
        self.дата = 0

    def получить(self):
        time.sleep(self.делэй)
        self.дата += 1
        return self.дата


class Камера():
    def __init__(self, разрешение, какая):
        части = разрешение.split('*')
        self.ширина = int(части[0])
        self.высота = int(части[1])
        # self.камера = cv2.VideoCapture(какая)

    def получить(self):
        # рет, кадр = self.камера.read()
        кадр = cv2.imread("hubble_cgcg396-2_potw2227a.jpg")
        return кадр

    # def __del__(self):
    #     self.камера.release()


def обработка_датчика(очередь, датчик):
    while (1):
        новое_значение = датчик.получить()
        while not очередь.empty():
            очередь.get()
        очередь.put(новое_значение)


def обработка_камеры(очередь, камера):
    ширина = камера.ширина
    высота = камера.высота
    while (1):
        новое_значение = камера.получить()
        while not очередь.empty():
            очередь.get()
        новое_значение = cv2.resize(новое_значение, (ширина, высота), interpolation=cv2.INTER_AREA)
        очередь.put(новое_значение)


def парсить(с1, с2, с3, ск):
    x = ск.shape[1] - 700  # Правая граница минус смещение по горизонтали
    y = ск.shape[0] - 50  # Нижняя граница минус смещение по вертикали

    шрифт = cv2.FONT_HERSHEY_SIMPLEX
    раземер_шрифта = 0.9
    толщина_шрифта = 2
    цвет_шрифта = (255, 255, 255)

    текст = f"Sensor 1: {с1} Sensor 2: {с2} Sensor 3: {с3}"

    cv2.putText(ск, текст, (x, y), шрифт, раземер_шрифта, цвет_шрифта, толщина_шрифта)

    return ск


def мэйн(аргументы):
    эщкере = 'эщкере'
    сенсор_1 = Сенсор(делэй=1)
    сенсор_2 = Сенсор(делэй=0.1)
    сенсор_3 = Сенсор(делэй=0.01)
    камера = Камера(аргументы.разрешение, аргументы.имя_камеры)

    очередь_1 = multiprocessing.Queue()
    очередь_2 = multiprocessing.Queue()
    очередь_3 = multiprocessing.Queue()
    очередь_камера = multiprocessing.Queue()

    процесс_1 = multiprocessing.Process(target=обработка_датчика, args=(очередь_1, сенсор_1))
    процесс_2 = multiprocessing.Process(target=обработка_датчика, args=(очередь_2, сенсор_2))
    процесс_3 = multiprocessing.Process(target=обработка_датчика, args=(очередь_3, сенсор_3))
    процесс_камера = multiprocessing.Process(target=обработка_камеры, args=(очередь_камера, камера))

    процесс_1.start()
    процесс_2.start()
    процесс_3.start()
    процесс_камера.start()

    ширина = камера.ширина
    высота = камера.высота
    старое_изображение = np.zeros((высота, ширина, 3), np.uint8)

    старый_1 = 0
    старый_2 = 0
    старый_3 = 0

    while (1):
        if not очередь_1.empty():
            старый_1 = очередь_1.get()

        if not очередь_2.empty():
            старый_2 = очередь_2.get()

        if not очередь_3.empty():
            старый_3 = очередь_3.get()

        if not очередь_камера.empty():
            старое_изображение = очередь_камера.get()

        картинка = парсить(старый_1, старый_2, старый_3, старое_изображение)
        cv2.imshow("Показания датчиков", картинка)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

    процесс_1.terminate()
    процесс_2.terminate()
    процесс_3.terminate()
    процесс_камера.terminate()

    процесс_1.join()
    процесс_2.join()
    процесс_3.join()
    процесс_камера.join()

    del сенсор_1
    del сенсор_2
    del сенсор_3
    del камера

    print(эщкере)


if __name__ == '__main__':
    парсер = argparse.ArgumentParser()
    парсер.add_argument('--имя_камеры', type=int, default=0, help='Имя камеры в системе')
    парсер.add_argument('--разрешение', type=str, default='900*900', help='Желаемое разрешение камеры')
    парсер.add_argument('--частота', type=int, default=40, help='Частота отображения картинки')
    аргументы = парсер.parse_args()
    мэйн(аргументы)
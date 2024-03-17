from multiprocessing import Process, Value, Array


def fun(number, array):
    number.value = 3.1415927
    for i in range(len(array)):
        array[i] = -array[i]


if __name__ == '__main__':
    num = Value('d', 0.0)
    arr = Array('i', range(10))

    proc = Process(target=fun, args=(num, arr,))
    proc.start()
    proc.join()

    print(num.value)
    print(arr[:])
    
import multiprocessing


def fun(queue):
    queue.put([42, None, 'hello'])


if __name__ == '__main__':
    queue = multiprocessing.Queue()

    proc = multiprocessing.Process(target=fun, args=(queue,))
    
    proc.start()
    print(queue.get())    # prints "[42, None, 'hello']"
    proc.join()
    
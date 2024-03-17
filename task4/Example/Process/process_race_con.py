import multiprocessing
import time


class Counter:
    def __init__(self):
        self._val = multiprocessing.Value('i',0)
  
  # change "self.val += 1" "self.val += int(1)"
    def change(self):
        with self._val.get_lock():
            self._val.value += int(1)

    def get_val(self):
        return self._val.value


# working with a shared variable
def work(counter: Counter, operationsCount):
    for _ in range(operationsCount):
        counter.change()


if __name__ == "__main__":  
    # shared variable
    counter = Counter()

    # start process
    process_list = []
    processCount = 10
    operationsPerProcessCount = 100000

    start = time.monotonic()
    for _ in range(processCount):
        proc = multiprocessing.Process(target=work, args=(counter, operationsPerProcessCount,))
        proc.start()
        process_list.append(proc)
  
    
    for proc in process_list:
        proc.join()

    duration = time.monotonic() - start

    expectedCounterValue = processCount * operationsPerProcessCount
    print(f"Expected val: {expectedCounterValue}, actual val: {counter.get_val()}")
    print(f'Duration: {round(duration, 4)}')
    
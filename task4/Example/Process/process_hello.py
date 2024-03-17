import multiprocessing

def fun_hello(name: str) -> None:
    print(f'Hello {name}!')


# можете убедиться, что это два различных процесса
# ps -eLo ppid,pid,tid,psr,args | grep process_hello.py
if __name__ == "__main__":
    name = "World"
    proc = multiprocessing.Process(target=fun_hello, args=(name,))
    proc.start()
    proc.join()
    
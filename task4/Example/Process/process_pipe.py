from multiprocessing import Process, Pipe


def fun(conn):
    conn.send([42, None, 'hello'])
    conn.close()


if __name__ == '__main__':
    parent_conn, child_conn = Pipe(duplex=True)
    proc = Process(target=fun, args=(child_conn,))
    proc.start()

    print(parent_conn.recv())   # prints "[42, None, 'hello']"
    proc.join()
    
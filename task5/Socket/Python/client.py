# Echo client program
import socket
import time

HOST = '10.9.43.13'    # Выбираем ip устройство в подсети 
PORT = 10011          # Выбираем порт устройства в подсети    
with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    # подключаемся к серверу
    s.connect((HOST, PORT))
    # Продолжаем отправлять данные, пока не будут отправлены все данные
    while 1:
        s.sendall(b'eskereeee')
        data = s.recv(1024)
        print('Received', repr(data))
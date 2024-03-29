# Echo server program
import socket

HOST = ''       # Назначаем ip сервера на устройстве, INADDR_ANY            
PORT = 10010    # Назначаем порт сервера на устройстве связанный с адресом HOST ip

# Создаем сокет, не забывайте закрывать сокет, если работаете без with
with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    # связываем сокет с заданным ip и port
    s.bind((HOST, PORT))
    # разрешаем подключения одного устройства
    s.listen(1)
    # устанавливаем подключение, и получаем сокет для общения с client
    conn, addr = s.accept()
    with conn:
        print('Connected by', addr)
        while True:
            data = conn.recv(1024)
            # сокет на другом конце закрылся
            if not data: break
            conn.sendall(data)
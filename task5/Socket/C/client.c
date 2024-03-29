#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <arpa/inet.h>

char message[] = "Hello there!\n";
// буфер для приема сообщения
char buf[sizeof(message)];

int main()
{

    int sock; // дескриптор сокета
    struct sockaddr_in addr;

    // создаем сокет клиента
    sock = socket(AF_INET, SOCK_STREAM, 0);
    if (sock < 0)
    {
        perror("socket");
        exit(1);
    }

    // конфигурируем адрес интернет сокета,
    // для подключения
    addr.sin_family = AF_INET;                     // семейство адресов
    addr.sin_port = htons(10010);                  // выбираем порт
    addr.sin_addr.s_addr = inet_addr("127.0.0.1"); // 127.0.0.1
    // подключаемся к серверу
    if (connect(sock, (struct sockaddr *)&addr, sizeof(addr)) < 0)
    {
        perror("connect");
        exit(2);
    }

    // отправляем сообщение
    send(sock, message, sizeof(message), 0);
    // принимаем сообщение
    recv(sock, buf, sizeof(message), 0);

    printf("%s", buf);
    
    //закрываем сокет
    close(sock);

    return 0;
}
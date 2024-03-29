#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <stddef.h>
#include <stdlib.h>
#include <unistd.h> 
#include <stdio.h>

int main()
{
    // дескриптор сокета для общения с клиентом,
    // и дескриптор сервера соответсвенно
    int sock, listener;
    // структура, описывающая адрес интернет-сокета
    struct sockaddr_in addr;
    // буфер для принятого сообщения
    char buf[1024];
    // количество принятых байт
    int bytes_read;

    // создаем сокет сервера
    listener = socket(AF_INET, SOCK_STREAM, 0);
    if(listener < 0)
    {
        perror("socket");
        exit(1);
    }
    
    // конфигурируем адрес интернет сокета,
    // для связывания с сокетом сервера
    addr.sin_family = AF_INET;
    addr.sin_port = htons(10010);
    addr.sin_addr.s_addr = htonl(INADDR_ANY); 
    // связываем
    if(bind(listener, (struct sockaddr *)&addr, sizeof(addr)) < 0)
    {
        perror("bind");
        exit(2);
    }

    // разрешаем подключения клиентов
    listen(listener, 1);
    
    while(1)
    {
        // ожидаем подключения
        sock = accept(listener, NULL, NULL);
        if(sock < 0)
        {
            perror("accept");
            exit(3);
        }

        while(1)
        {
            bytes_read = recv(sock, buf, 1024, 0);
            if(bytes_read <= 0) break;
            printf("%s", buf);
            send(sock, buf, bytes_read, 0);
        }
        // закрываем сокет
        close(sock);
    }
    
    return 0;
}
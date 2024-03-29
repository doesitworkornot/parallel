cmake_minimum_required(VERSION 3.22)

project(Socket_C VERSION 1.0 LANGUAGES C)

set(CMAKE_C_COMPILER "gcc-11")
set(CMAKE_C_STANDARD 17)
set(CLIENT_NAME "client")
set(SERVER_NAME "server")

message("Compile C: " ${CMAKE_C_COMPILER})

# Собираем сервер
add_executable(${SERVER_NAME} "server.c") # cоздаем исполняемый файл
target_compile_options(${SERVER_NAME} PRIVATE -O2) # опции

# Собираем клиент
add_executable(${CLIENT_NAME} "client.c") # cоздаем исполняемый файл
target_compile_options(${CLIENT_NAME} PRIVATE -O2) # опции


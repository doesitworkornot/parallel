cmake_minimum_required(VERSION 3.22)

project(Socket_asio_Boost VERSION 1.0 LANGUAGES CXX)

set(CMAKE_CXX_COMPILER "g++-11")
set(CMAKE_CXX_STANDARD 20)
set(CLIENT_NAME "boost_client")
set(SERVER_NAME "boost_server")

message("Compile C++: " ${CMAKE_CXX_COMPILER})

find_package(Boost COMPONENTS system)

if(Boost_FOUND AND Boost_SYSTEM_LIBRARY)
    # Собираем сервер
    add_executable(${SERVER_NAME} "server_asio.cpp") # cоздаем исполняемый файл
    target_include_directories(${SERVER_NAME} PRIVATE ${Boost_INCLUDE_DIR}) # подключаем заголовочные файлы
    target_compile_options(${SERVER_NAME} PRIVATE -O2) # опции
    target_link_libraries(${SERVER_NAME} ${Boost_LIBRARIES}) # подключаем библиотеку
    
    # Собираем клиент
    add_executable(${CLIENT_NAME} "client_asio.cpp") # cоздаем исполняемый файл
    target_include_directories(${CLIENT_NAME} PRIVATE ${Boost_INCLUDE_DIR}) # подключаем заголовочные файлы
    target_compile_options(${CLIENT_NAME} PRIVATE -O2) # опции
    target_link_libraries(${CLIENT_NAME} ${Boost_LIBRARIES}) # подключаем библиотеку
else()
    message("Boost or library boost_system not found!")
endif()

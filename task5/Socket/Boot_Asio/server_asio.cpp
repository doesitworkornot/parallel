#include <iostream>
#include <boost/asio.hpp>
#include <string>

namespace ip_boost = boost::asio::ip;

// Читаем сообщение из сокета
std::string read_(ip_boost::tcp::socket & socket) {
       boost::asio::streambuf buf;
       boost::asio::read_until( socket, buf, "\n" );
       std::string data = boost::asio::buffer_cast<const char*>(buf.data());
       return data;
}

// Отправляем сообщение в сокет
void send_(ip_boost::tcp::socket & socket, const std::string& message) {
       const std::string msg = message + "\n";
       boost::asio::write( socket, boost::asio::buffer(message) );
}

int main() {
      boost::asio::io_service io_service;
//listen for new connection
      ip_boost::tcp::acceptor acceptor_(io_service, ip_boost::tcp::endpoint(ip_boost::tcp::v4(), 10010 ));
//socket creation 
      ip_boost::tcp::socket socket_(io_service);
//waiting for connection
      acceptor_.accept(socket_);
//read operation
      std::string message = read_(socket_);
      std::cout << message << std::endl;
//write operation
      send_(socket_, "Hello From Server!");
      std::cout << "Servent sent Hello message to Client!" << std::endl;
   return 0;
}

main: dgem

dgem: mk
	g++ -fopenmp vecmx.cpp -o build/dgem

clse: mk
	g++ -pthread client_serv.cpp -o build/client_serv

mk:
	mkdir -p build
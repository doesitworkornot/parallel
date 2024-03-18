main: dgem

dgem: mk
	g++ -fopenmp vecmx.cpp -o build/dgem

clse: mk
	g++ -pthread client_serv.cpp -o build/client_serv

test: mk
	g++ test.cpp -o build/test

mk:
	mkdir -p build
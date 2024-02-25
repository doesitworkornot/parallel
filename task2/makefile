main: dgem

dgem: mk
	gcc -fopenmp DGEMV.c -o build/dgem

integral: mk
	gcc -fopenmp integral.c -o build/integral -lm

sle: mk
	gcc -fopenmp sle.c -o build/sle -lm

mk:
	mkdir -p build

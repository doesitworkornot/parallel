LD=pgc++
LDFLAGS=-fast -ta=tesla:managed 
NVTXLIB := -I/opt/nvidia/hpc_sdk/Linux_x86_64/20.9/cuda/11.0/include
boot = -I/path/to/boost/include -L/path/to/boost/lib -lboost_program_options


exe:
	$(LD) $(LDFLAGS) -o $@ laplace2d.cpp jacobi.cpp ${NVTXLIB} ${boot}

all: exe run


run: exe
	./exe

.PHONY: clean

clean:
	-rm -f *.o *.mod core exe


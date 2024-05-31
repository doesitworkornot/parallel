CXX=nvc++
LD=nvc++
CXXFLAGS=-std=c++11 -fast -ta=tesla:managed
LDFLAGS=-fast -ta=tesla:managed

NVTXLIB := -I/opt/nvidia/hpc_sdk/Linux_x86_64/20.9/cuda/11.0/include -L/usr/local/cuda/lib64 -cudalib=cublas
boot = -I/path/to/boost/include -L/path/to/boost/lib -lboost_program_options

all: exe

exe: jacobi.o
	$(LD) $(LDFLAGS) ${NVTXLIB} -o $@ $^ ${boot}

.PHONY: clean
clean:
	-rm -f *.o *.mod core exe
.SUFFIXES: .c .cpp .F90 .o
.cpp.o:
	$(CXX) $(CXXFLAGS) -c -o $@ $< ${NVTXLIB} ${boot}

CC = g++
CUDA_PATH = /usr/local/cuda
#CUDA_PATH = /pub/cuda-7.5
NVCC = $(CUDA_PATH)/bin/nvcc -ccbin $(CC)
CFLAGS = -L$(CUDA_PATH)/lib64 -lcudart -lcuda -lcurand -lm  
NVCCFLAGS= -D_FORCE_INLINES -Xcompiler -fPIC  -I$(CUDA_SDK_PATH)/samples/common/inc -I/usr/local/cuda/include/
COPTFLAGS = -O3 -g -std=c++11 
LDFLAGS =

SMS ?= 30 35 37 50 52 60 70 

ifeq ($(GENCODE_FLAGS),)                                                        
$(foreach sm,$(SMS),$(eval GENCODE_FLAGS += -gencode arch=compute_$(sm),code=sm_$(sm)))
endif 

all: 	main.o main_cpu.o world.o common.o ga_cpu.o ga_gpu.o tsp-ga-gpu tsp-ga-cpu

tsp-ga-gpu: main.o 
	$(NVCC) main.o world.o ga_gpu.o common.o  $(CFLAGS) $(COPTFLAGS) -o tsp-ga-gpu
tsp-ga-cpu: main_cpu.o 
	$(NVCC) main_cpu.o world.o ga_cpu.o common.o  $(CFLAGS) $(COPTFLAGS) -o tsp-ga-cpu

ga_gpu.o: ga_gpu.cu  world.h
	$(NVCC) $(GENCODE_FLAGS) -c ga_gpu.cu $(CFLAGS) $(COPTFLAGS) $(NVCCFLAGS)


ga_cpu.o: ga_cpu.cpp  world.h common.h
	$(NVCC) -c ga_cpu.cpp $(CFLAGS) $(COPTFLAGS)

common.o: common.cpp common.h  world.h
	$(NVCC) -c common.cpp $(CFLAGS) $(COPTFLAGS)

world.o: world.cpp world.h
	$(NVCC) -c world.cpp $(CFLAGS) $(COPTFLAGS) $(NVCCFLAGS)
main.o: main.cpp  common.h ga_gpu.h  world.h
	$(NVCC) -c main.cpp $(CFLAGS) $(COPTFLAGS) $(NVCCFLAGS)
main_cpu.o: main_cpu.cpp  common.h ga_cpu.h  world.h
	$(NVCC) -c main_cpu.cpp $(CFLAGS) $(COPTFLAGS) 

.PHONY: clean
clean:
	rm -rf *.o tsp-ga-cpu tsp-ga-gpu

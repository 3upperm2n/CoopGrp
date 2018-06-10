CC = g++
CUDA_PATH = /usr/local/cuda
#CUDA_PATH = /pub/cuda-7.5
NVCC = $(CUDA_PATH)/bin/nvcc -ccbin $(CC)
CFLAGS = -L$(CUDA_PATH)/lib64 -lcudart -lcuda -lm  -lcurand -lcurand_static -lculibos
NVCCFLAGS= -I$(CUDA_PATH)/samples/common/inc -I$(CUDA_PATH)/include -dc -D_FORCE_INLINES -Xcompiler -fPIC  
COPTFLAGS = -O3 -g -std=c++11 
LDFLAGS =

#SMS ?= 30 35 37 50 52 60 
SMS ?= 70 

ifeq ($(GENCODE_FLAGS),)                                                        
$(foreach sm,$(SMS),$(eval GENCODE_FLAGS += -gencode arch=compute_$(sm),code=sm_$(sm)))
endif 

all: 	main.o main_cpu.o world.o common.o ga_cpu.o ga_gpu.o tsp-ga-gpu tsp-ga-cpu

tsp-ga-gpu: main.o 
	$(NVCC) $(GENCODE_FLAGS) main.o world.o ga_gpu.o common.o  $(CFLAGS) $(COPTFLAGS) -o tsp-ga-gpu
tsp-ga-cpu: main_cpu.o 
	$(NVCC) $(GENCODE_FLAGS) main_cpu.o world.o ga_cpu.o common.o  $(CFLAGS) $(COPTFLAGS) -o tsp-ga-cpu

ga_gpu.o: ga_gpu.cu  world.h
	$(NVCC) $(NVCCFLAGS) $(GENCODE_FLAGS) $(CFLAGS) $(COPTFLAGS) -c ga_gpu.cu 


ga_cpu.o: ga_cpu.cpp  world.h common.h
	$(NVCC) $(GENCODE_FLAGS) -c ga_cpu.cpp $(CFLAGS) $(COPTFLAGS)

common.o: common.cpp common.h  world.h
	$(NVCC) $(GENCODE_FLAGS) -c common.cpp $(CFLAGS) $(COPTFLAGS)

world.o: world.cpp world.h
	$(NVCC) $(GENCODE_FLAGS) -c world.cpp $(CFLAGS) $(COPTFLAGS) $(NVCCFLAGS)

main.o: main.cu common.h ga_gpu.h  world.h
	$(NVCC) $(NVCCFLAGS) $(GENCODE_FLAGS) -c main.cu $(CFLAGS) $(COPTFLAGS)

main_cpu.o: main_cpu.cpp  common.h ga_cpu.h  world.h
	$(NVCC) $(GENCODE_FLAGS) -c main_cpu.cpp $(CFLAGS) $(COPTFLAGS) 

.PHONY: clean
clean:
	rm -rf *.o tsp-ga-cpu tsp-ga-gpu

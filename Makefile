CU_CC=nvcc
CU_FLAGS=-O3 -arch compute_20 -Xcompiler -fopenmp
CU_LIBS=

CC=g++
CC_FLAGS=-O3
CC_LIBS=-fopenmp -lcudart -lstdc++ 
CC_CUL=/share/apps/cuda/lib64
CC_CUI=/share/apps/cuda/include


SOURCE=clbm_ldc3Dr7omp.cpp
TARGET=clbm_ldc3D

$(TARGET): $(SOURCE) lbm_utils.o lbm_functions_cuda.o lbm_functions.o ldc_3D_LBGK_tsR2.o
	$(CC) -o $(TARGET) $(SOURCE) $(CC_FLAGS) -L$(CC_CUL) -I$(CC_CUI) lbm_utils.o lbm_functions_cuda.o lbm_functions.o ldc_3D_LBGK_tsR2.o  $(CC_LIBS)

lbm_utils.o:  lbm_utils.cpp
	$(CC) -c lbm_utils.cpp -O3 

lbm_functions_cuda.o: lbm_functions_cuda.cu
	$(CU_CC) -c lbm_functions_cuda.cu $(CU_FLAGS) 

ldc_3D_LBGK_tsR2.o:  ldc_3D_LBGK_tsR2.cu
	$(CU_CC) -c ldc_3D_LBGK_tsR2.cu $(CU_FLAGS)

lbm_functions.o: lbm_functions.cpp
	$(CC) -c lbm_functions.cpp -O3 -fopenmp


clean:
	rm *.o



commons = data.h parser.h parser.c

clean:
	rm *.out 

test1:
	gcc $(commons) unitTest_parser.c -o test_parser.out 

test2:
	gcc $(commons) unitTest_parser2.c -o test2_parser.out

cpu:
	gcc $(commons) main_cpu.c -lm -O3 -o cnn_cpu.out 

cpu_debug:
	gcc $(commons) main_cpu_debug.c -lm -g -o cnn_cpu_db.out 

cpu_debug_exe:
	gcc $(commons)  main_cpu.c -lm -g -o cnn_cpu_db_exe.out 

gpu:
	nvcc $(commons) main_gpu.c basic_gpu.cu -lm -O3 -o cnn_gpu_basic.out 

gpu_tile:
	nvcc $(commons) main_gpu.c tile_gpu.cu -lm -O3 -o cnn_gpu_tile.out 

gpu_matrix:
	nvcc $(commons) main_gpu.c matrix_gpu.cu -lm -O3 -o cnn_gpu_matrix.out 

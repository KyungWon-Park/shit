clean:
	rm *.out 

test1:
	gcc data.h parser.h parser.c unitTest_parser.c -o test_parser.out 

test2:
	gcc data.h parser.h parser.c unitTest_parser2.c -o test2_parser.out

cpu:
	gcc data.h parser.h parser.c main_cpu.c -lm -O3 -o cnn_cpu.out 

cpu_debug:
	gcc data.h parser.h parser.c main_cpu_debug.c -lm -g -o cnn_cpu_db.out 

cpu_debug_exe:
	gcc data.h parser.h parser.c main_cpu.c -lm -g -o cnn_cpu_db_exe.out 

gpu:

gpu_debug:

gpu_tile:

gpu_tile_debug:

gpu_matrix:

gpu_matrix_debug:

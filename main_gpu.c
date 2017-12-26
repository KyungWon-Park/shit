#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "parser.h"

#ifdef BASIC
#include "basic_gpu.cu"
#endif 

#ifdef TILE
#include "tile_gpu.cu"
#endif 

#ifdef MATRIX
#include "matrix_gpu.cu"
#endif

const char *PATH_TEST_DATA = "./mnist/t10k-images-idx3-ubyte";
const char *PATH_TEST_LABEL= "./mnist/t10k-labels-idx1-ubyte";

#ifdef DEBUG 
void prompt(void)
{
	printf("Press ENTER key to continue\n");
	char asdf;
	fgets(&asdf, 100, stdin);
	return;
}
#endif

int main(int argc, char *argv[])
{// main entry point
	__map__ map;
	load_weights(&map);

	// Read MNIST data set
	// WANRING: MALLOC 0
	float *test_data = malloc(sizeof(float) * NUM_TEST * 32 * 32);
	int *test_label = malloc(sizeof(int) * NUM_TEST);
	read_data(PATH_TEST_DATA, test_data);
	read_label(PATH_TEST_LABEL, test_label);

	// Variables
	int cnt_correct = 0;

#ifndef DEBUG 
	clock_t start = clock(), diff;
#endif 

	forward_GPU(&test_data, &test_label, &map, &cnt_correct);

#ifndef DEBUG 
	diff = clock() - start;
	int msec = diff * 1000 / CLOCKS_PER_SEC;
	printf("Elapsed time: %d seconds %d milliseconds\n", msec / 1000, msec % 1000);
#endif 
	printf("Accuracy: %f %%\n", ((float) cnt_correct / ((float) NUM_TEST) * 100));

	// WARNING: FREE 0
	free(test_data);
	free(test_label);
	return 0;
}



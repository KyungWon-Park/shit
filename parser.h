#include "data.h"
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

#define NUM_TRAIN 60000
#define NUM_TEST  10000
#define RAW_DIM   28
#define RAW_PIXELS_PER_IMG 784 			// 28x28, single channel image
#define MNIST_SCALE_FACTOR 0.00390625	// 1/255
#define MAXBYTE 255 

void load_weights(__map__ *ptr_map);
void read_data(const char *datapath, float *data);
void read_label(const char *labelPath, int *labels);
void printMNIST(float *data, int label);

//#include <cuda.h>
#include <string.h>
#include <math.h>
#include "parser.h"

#define BATCH_SIZE 32

/*
   * # of SM: 28 
   * # of cores per SM: 128
   * Max # of threads per SM: 2048
   * Max # of threads per block: 1024
   * shared memory per block: 49152 bytes;
   */

__constant__ int D_BATCH_SIZE;
__constant__ float d_test_data[NUM_TEST];
__constant__ __map__ d_map;
// test_data of MNIST in Device constant memory 

__global__ void convolution_kernel(int curr_step, int num_input, int num_output, int height_input, int width_input, int size_filter, int stage)
{// Convolution computation kernel
	// curr_step: Which step are we in? Judge input based on 'curr_step' and 'D_BATCH_SIZE'
	// num_input: number of input channels
	// num_output: number of output chanels
	// height_input: height of input image
	// width_input: width of input image
	// size_filter: size of filter map 
	// stage: which stage are we in? (1 means C1) or (3 means C3) 

	int tid = threadIdx.x;
	int bid = blockIdx.x;
	int bsize = blockDim.x;
	
	__shared__ filter_C1[6][1][5][5];
	__shared__ filter_C3[16][6][5][5];
	if (stage == 1)
	{
		if (tid < 30)
		{
			for (int i = 0; i < 5; i++)
			{
				filter[tid / 5][1][tid % 5][i] = d_map.C1_param[tid / 5][1][tid % 5][i];
			}
		}
		float *output = d_c1_results;
	}
	else 
	{
		float *output = d_c3_results;
	}
	
	return;
}

__global__ void pooling_kernel()
{
	return;
}

__global__ void fullyConnect_kernel()
{
	return;
}

__global__ void output_kernel()
{
	return;
}

void forward_GPU(float **ptr_test_data, int **ptr_test_label, __map__ *map, int *cnt_correct)
{// Deploy forward computation job on GPU
	float *test_data = *ptr_test_data;
	int *test_label = *ptr_test_label;

	// Acquire memory space in GPU 
	// Prefix "d_" means ADDRESS in device memory 
	int inferences[BATCH_SIZE];
	int *d_inferences;

	float *d_c1_results;
	float *d_s2_results;
	float *d_c3_results;
	float *d_s4_results;
	float *d_f5_results;
	float *d_f6_results;
	float *d_output_results;
	// WARNING: MALLOC 0
	cudaMalloc((void **) &d_inferences, sizeof(int) * BATCH_SIZE);
	cudaMalloc((void **) &d_c1_results, sizeof(float) * BATCH_SIZE * 6 * 28 * 28);
	cudaMalloc((void **) &d_s2_results, sizeof(float) * BATCH_SIZE * 6 * 14 * 14);
	cudaMalloc((void **) &d_c3_results, sizeof(float) * BATCH_SIZE * 16 * 10 * 10);
	cudaMalloc((void **) &d_s4_results, sizeof(float) * BATCH_SIZE * 16 * 5 * 5);
	cudaMalloc((void **) &d_f5_results, sizeof(float) * BATCH_SIZE * 120);
	cudaMalloc((void **) &d_f6_results, sizeof(float) * BATCH_SIZE * 84);
	cudaMalloc((void **) &d_output_results, sizeof(float) * BATCH_SIZE * 10);

	// CUDA memcpy from host to device 
	int D_NUM_TEST = ((int) ceil((double) ((float) NUM_TEST / (float) BATCH_SIZE))) * BATCH_SIZE;
	int dummy = BATCH_SIZE;
	cudaMemcpyToSymbol(d_test_data, test_data, sizeof(float) * D_NUM_TEST * 32 * 32, 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(D_BATCH_SIZE, &dummy, sizeof(int), 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(d_map, map, sizeof(__map__), 0, cudaMemcpyHostToDevice);
	cudaMemcpy(d_inferences, inferences, sizeof(int) * BATCH_SIZE, cudaMemcpyHostToDevice);

	// ENTERING MAIN LOOP
	int step = 0;
	for (int step = 0; (step * BATCH_SIZE) < D_NUM_TEST; step++)
	{// Advance step by step, with BATCH_SIZE stride
		// 0. Convolution layer C1

		// 1. Pooling layer S2 

		// 2. Convolution layer C3

		// 3. Pooling layer S4

		// 4. Fully connected layer F5

		// 5. Fully connected layer F6

		// 6. Output layer OUTPUT

		// 7. Update cnt_correct
		cudaMemcpy(inferences, d_inferences, sizeof(int) * BATCH_SIZE, cudaMemcpyDeviceToHost);
		for (int i = 0; i < BATCH_SIZE; i++)
		{// For every result numbers in BATCH
			int index = (step * BATCH_SIZE) + i;
			if (index >= NUM_TEST)
			{// Check that our BATCH didn't go out of NUM_TEST 
				break;
			}
			else 
			{// If this inferences[i] is valid result, 
				if (inferences[i] == test_label[index])
				{// If such inferences[i] is same with test_label[index]
					(*cnt_correct)++;
				}
			}
		}
	}

	// WARNING: FREE 0
	cudaFree(d_inferences);
	cudaFree(d_c1_results);
	cudaFree(d_s2_results);
	cudaFree(d_c3_results);
	cudaFree(d_s4_results);
	cudaFree(d_f5_results);
	cudaFree(d_f6_results);
	cudaFree(d_output_results);
	
	return;
}

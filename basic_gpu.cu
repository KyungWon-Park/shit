#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <string.h>
#include <math.h>
#include <assert.h>
#include "parser.h"

#define BATCH_SIZE 32 
/* BATCH_SIZE 
   *
   * Why use BATCH_SIZE ?
   *
   * 0. Saturate Streaming Multiprocessors with enough computaion BLOCKS
   * 1. Saturate Video RAM with enough computaional jobs
   * 
   * CRITERIA:
   * 	- Deploy enough blocks (More than n * SM counts) for latency hiding
   * 	- Saturate each block with enough threads 
   */

/* 				NVIDIA GEFORCE GTX1080
   * GPU SPEC:
   * 	- warp_size: 32 threads
   * 	- word_size: 4 Bytes
   * 	- SM_count: 20 Streaming Multiprocessors
   * 	
   * SM SPEC: 
   * 	- max_warps: 64
   * 	- max_thread_blocks : 32 
   * 	- max_threads: 2048
   * 	- max_registers: 65536 words
   * 	- CUDA_cores: 64 cores 
   * 	- share_memory: 64 kB
   *
   * BLOCK SPEC:
   * 	- max_threads: 1024
   * 	- max_registers: 65536 words
   *
   * THREAD SPEC:
   * 	- max_registers: 255 words 
   *
   */

/* Memory design 
 * 
 * 0. INPUT image data 
 * 	=> ALL goes into global memory
 *
 * 1. Filter map data 
 * 	=> Put as much as we can into constant memory (d_map), but leftover should go to global memory (d_map_spill)
 *
 * 2. Result data 
 * 	=> Should go to global memory since write-once
 *
 * 3. What to cache into shared memory?
 * 	=> Bring Filter map data into shared_memory (only necessary part)
 * 	=> Bring INPUT data into shared_memory (only necessary part)
 *
 */

__constant__ int D_BATCH_SIZE;
__constant__ int D_NUM_TEST;
__constant__ __gpu_map__ d_map;

__device__ float sigmoid(float x)
{
	return (1 / (1 + exp(-x)));
}

/*
   * ARGUMENTS:
   * 	- curr_step: Which step are we in? (In MAIN_LOOP)
   * 	- stage: Stage number(ex; 1 means C1 layer, 3 means C3 layer)
   * 	- num_output: Number of output maps 
   * 	- num_input: Number of input maps 
   * 	- height_input: Height of input maps 
   * 	- width_input: Width of input maps 
   * 	- size_filter: Size of filter map, 5 for LeNet-5
   * 	- d_map + d_map_spill: Contains filter maps for all layers
   * 	- inputs: Source of input images 
   * 	- outputs: Destination to store output(computed) images
   * 	- size_input: Length of input 1D array, for fully connected layer
   * 	- size_output: Length of output 1D array, for fully connected layer
   */

__global__ void 	// Convolution computation kernel  
convolution_kernel(
	int curr_step, int stage,
	int num_output, int num_input, int height_input, int width_input,
	int size_filter, __gpu_map__ *d_map,
	float *inputs, float *outputs
)
{
	int BID_x = blockIdx.x; 	// foreach: output image 	~6 or ~16 
	int BID_y = blockIdx.y; 	// foreach: BATCH among curr_step_inputs[BATCH_SIZE] 
	int TID_x = threadIdx.x; 	// foreach: output image row 	~28 or ~10 
	int TID_y = threadIdx.y; 	// foreach: output image column 	~28 or ~10

	float acc = 0;
	if (stage == 1)
	{// C1_layer convolution: D_BATCH_SIZE * { [1 @ 32 * 32] .X [6 * 1 @ 5 * 5] => [6 @ 28 * 28] }
		// Get the starting point from entire MNIST data set 
		float *input_start = inputs + (curr_step * D_BATCH_SIZE * (32 * 32)) + (BID_y * 32 * 32);

		// Load data into shared memory
		__shared__ float input[32][32];
		for (int i = 0; i < 2; i++)
		{
			int rp = 28 * i + TID_x;
			int cp = 28 * i + TID_y;
			if (rp < height_input && cp < width_input)
			{
				input[rp][cp] = input_start[(32 * rp) + cp];
			}
		}
		__syncthreads();
		__shared__ float filter[5][5];
		if (TID_x < size_filter && TID_y < size_filter) 
		{
			filter[TID_x][TID_y] = (*d_map).C1_param[BID_x][0][TID_x][TID_y]; 
		}
		__syncthreads();

		for (int f_row = 0; f_row < size_filter; f_row++)
		{
			for (int f_col = 0; f_col < size_filter; f_col++)
			{
				acc += input[TID_x + f_row][TID_y + f_col] * filter[f_row][f_col];
			}
		}
		outputs[(BID_y * 6 * 28 * 28) + (BID_x * 28 * 28) + (TID_x * 28) + TID_y] = acc; 
	}
	else // Desired stage = 3
	{// C3_layer convolution: D_BATCH_SIZE * { [6 @ 14 * 14] .X [16 * 6 @ 5 * 5] => [16 @ 10 * 10] }
		// Get the starting point from d_s2_results[BATCH_SIZE]
		float *input_start = inputs + (BID_y * (14 * 14));
		
		for (int c = 0; c < num_input; c++)
		{
			// Load data into shared memory 
			__shared__ float input[14][14];
			for (int i = 0; i < 2; i++)
			{
				int rp = 14 * i + TID_x;
				int cp = 14 * i + TID_y;
				if (rp < height_input && rp < width_input)
				{
					input[rp][cp] = input_start[(32 * rp) + cp];
				}
			}
			__syncthreads();
			__shared__ float filter[5][5];
			if (TID_x < size_filter && TID_y < size_filter)
			{
				filter[TID_x][TID_y] = (*d_map).C3_param[BID_x][c][TID_x][TID_y];
			}
			__syncthreads();

			for (int f_row = 0; f_row < size_filter; f_row++)
			{
				for (int f_col = 0; f_col < size_filter; f_col++)
				{
					acc += input[TID_x + f_row][TID_y + f_col] * filter[f_row][f_col];
				}
			}
		}
		outputs[(BID_y * 16 * 10 * 10) + (BID_x * 10 * 10) + (TID_x * 10) + TID_y];
	}

	return;
}

__global__ void 	// Pooling computation kernel
pooling_kernel(
	int curr_step, int stage,
	int num_output, int height_input, int width_input,
	__gpu_map__ *d_map,
	float *inputs, float *outputs 
)
{
	if (stage == 2)
	{// S2_layer pooling: D_BATCH_SIZE * { Sigmoid([6 @ 28 * 28] + bias[6]) => [6 @ 14 * 14] }
	}
	else // Desired stage = 4
	{// S4_layer pooling: D_BATCH_SIZE * { Sigmoid([16 @ 10 * 10] + bias[16]) => [16 @ 5 * 5] }
	}
	return;
}

__global__ void 	// Fully connecting computation kernel 
fullyConnect_kernel(
	int curr_step, int stage,
	int size_input, int size_output,
	__gpu_map__ *d_map, __gpu_map_spill__ *d_map_spill,
	float *inputs, float *outputs 
)
{
	if (stage == 5)
	{// F5_layer full connection: D_BATCH_SIZE * { Sigmoid([120 * 400] X Serial[16 @ 5 * 5] + bias[120 * 1]) => [120 * 1] }
	}
	else // Desired stage = 6
	{// F6_layer full connection: D_BATCH_SIZE * { Sigmoid([84 * 120] X [120 * 1] + bias[84 * 1]) => [84 * 1] }
	}
	return;
}

__global__ void 	// Output layer compuation kernel 
output_kernel(
	int curr_step, int stage,
	int size_input, int size_output,
	__gpu_map__ *d_map, __gpu_map_spill__ *d_map_spill,
	float *inputs, float *outputs
)
{
	// OUTPUT_layer: D_BATCH_SIZE * { [10 * 84] X [84 * 1] + [10 * 1] => [10 * 1] }
	return;
}

__global__ void 	// Number determination kernel 
numberDetermine_kernel(
	int curr_step, int stage,
	float *inputs, int *outputs
)
{
	// NUMBER_layer: D_BATCH_SIZE * { ReduceMax[10 * 1] => SINGLE_DIGIT }
	return;
}

void forward_GPU(float **ptr_test_data, int **ptr_test_label, __map__ *map, int *cnt_correct)
{// Deploy forward computation job on GPU
	float *test_data = *ptr_test_data;
	int *test_label = *ptr_test_label;

	// Acquire memory space in GPU 
	// Prefix "d_" means ADDRESS in device memory 
	// Handlers for device memory manipulation
	int inferences[BATCH_SIZE];
	int *d_inferences;

	float *d_test_data;
	__gpu_map_spill__ *d_map_spill;

	float *d_c1_results;
	float *d_s2_results;
	float *d_c3_results;
	float *d_s4_results;
	float *d_f5_results;
	float *d_f6_results;
	float *d_output_results;

	// WARNING: MALLOC 1
	__gpu_map__ *tmp_map = malloc(sizeof(__gpu_map__));
	__gpu_map_spill__ *tmp_map_spill = malloc(sizeof(__gpu_map_spill__));
	assert(tmp_map != NULL && "MALLOC FAILED!\n");
	assert(tmp_map_spill != NULL && "MALLOC FAILED!\n");

	// Fill in gpu_map data
	// tmp_map = map - F5_param 
	memcpy((*tmp_map).C1_param, (*map).C1_param, sizeof(float) * 6 * 1 * 5 * 5);
	memcpy((*tmp_map).C1_bias, (*map).C1_bias, sizeof(float) * 6);
	memcpy((*tmp_map).C3_param, (*map).C3_param, sizeof(float) * 16 * 6 * 5 * 5);
	memcpy((*tmp_map).C3_bias, (*map).C3_bias, sizeof(float) * 16);
	memcpy((*tmp_map).F5_bias, (*map).F5_bias, sizeof(float) * 120);
	memcpy((*tmp_map).F6_param, (*map).F6_param, sizeof(float) * 84 * 120);
	memcpy((*tmp_map).F6_bias, (*map).F6_bias, sizeof(float) * 84);
	memcpy((*tmp_map).OUTPUT_param, (*map).OUTPUT_param, sizeof(float) * 10 * 84);
	memcpy((*tmp_map).OUTPUT_bias, (*map).OUTPUT_bias, sizeof(float) * 10);

	// tmp_map_spill = F5 param
	memcpy((*tmp_map_spill).F5_param, (*map).F5_param, sizeof(float) * 120 * 400);

	// Fix NUM_TEST into d_NUM_TEST so d_NUM_TEST can be multiple of BATCH_SIZE, so we can walk in stride
	int d_NUM_TEST = ((int) ceil((double) ((float) NUM_TEST / (float) BATCH_SIZE))) * BATCH_SIZE;
	int batch_size = BATCH_SIZE;

	// WARNING: MALLOC 0
	cudaMalloc((void **) &d_inferences, sizeof(int) * BATCH_SIZE);
	cudaMalloc((void **) &d_test_data, sizeof(float) * d_NUM_TEST * 32 * 32);
	cudaMalloc((void **) &d_map_spill, sizeof(__gpu_map_spill__));
	cudaMalloc((void **) &d_c1_results, sizeof(float) * BATCH_SIZE * 6 * 28 * 28);
	cudaMalloc((void **) &d_s2_results, sizeof(float) * BATCH_SIZE * 6 * 14 * 14);
	cudaMalloc((void **) &d_c3_results, sizeof(float) * BATCH_SIZE * 16 * 10 * 10);
	cudaMalloc((void **) &d_s4_results, sizeof(float) * BATCH_SIZE * 16 * 5 * 5);
	cudaMalloc((void **) &d_f5_results, sizeof(float) * BATCH_SIZE * 120);
	cudaMalloc((void **) &d_f6_results, sizeof(float) * BATCH_SIZE * 84);
	cudaMalloc((void **) &d_output_results, sizeof(float) * BATCH_SIZE * 10);

	// CUDA memcpy from host to device 
	cudaMemcpyToSymbol(D_NUM_TEST, &d_NUM_TEST, sizeof(int), 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(D_BATCH_SIZE, &batch_size, sizeof(int), 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(d_map, tmp_map, sizeof(__gpu_map__), 0, cudaMemcpyHostToDevice);
	cudaMemcpy(d_map_spill, tmp_map_spill, sizeof(__gpu_map_spill__), 0, cudaMemcpyHostToDevice);

	// WARNING: FREE 1
	free(tmp_map);
	free(tmp_map_spill);

	// ENTERING MAIN LOOP
	int step = 0;
	dim3 block;
	dim3 thread;
	for (int step = 0; (step * BATCH_SIZE) < d_NUM_TEST; step++)
	{// Advance step by step, with BATCH_SIZE stride 
		// START
		// 0. Convolution layer C1

		// 1. Pooling layer S2 

		// 2. Convolution layer C3

		// 3. Pooling layer S4

		// 4. Fully connected layer F5

		// 5. Fully connected layer F6

		// 6. Output layer OUTPUT

		// 7. Determine number 

		// 8. Update cnt_correct
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
				{// If such inferences[i] is same with test_label[index], increment cnt_correct counter
					(*cnt_correct)++;
				}
			}
		}
	}

	// WARNING: FREE 0
	cudaFree(d_inferences);
	cudaFree(d_map_spill);
	cudaFree(d_test_data);
	cudaFree(d_c1_results);
	cudaFree(d_s2_results);
	cudaFree(d_c3_results);
	cudaFree(d_s4_results);
	cudaFree(d_f5_results);
	cudaFree(d_f6_results);
	cudaFree(d_output_results);
	
	return;
}

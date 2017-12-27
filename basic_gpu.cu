#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <string.h>
#include <math.h>
#include <assert.h>
#include "parser.h"

#define BATCH_SIZE 20
/* BATCH_SIZE 
   *
   * Why use BATCH_SIZE ? (Multiple images at once)
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
   * SHARED MEMORY SPEC:
   * 	- 64 kB per SM 
   * 	- Composed of 32 memory bank hardwares
   * 	- Does bank interleaving per every word (4 Bytes)
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
	int size_filter,
	float *inputs, float *outputs
)
{
	// Get index info 
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
		// Do shared memory access in 32 stride to avoid shared memory bank conflict
		int myCnt = 28 * TID_x + TID_y;
		if (myCnt < 32)
		{
			for (int i = 0; i < 32; i++)
			{
				input[i][myCnt] = input_start[(32 * i) + myCnt]; 
			}
		}
		__syncthreads();
		__shared__ float filter[5][5]; 	// Only 25 entries -> No shared memory bank conflict 
		if (TID_x < size_filter && TID_y < size_filter) 
		{
			filter[TID_x][TID_y] = d_map.C1_param[BID_x][0][TID_x][TID_y]; 
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
		{// For every input channel, which isn't 1 for C3 layer
			// Load data into shared memory 
			__shared__ float input[14][14];
			// Do shared memory access in 14 strides to avoid shared memory bank conflict 
			int myCnt = 10 * TID_x + TID_y;
			if (myCnt < 14)
			{
				for (int i = 0; i < 14; i++)
				{
					input[i][myCnt] = input_start[(14 * i) + myCnt];
				}
			}
			__syncthreads();
			__shared__ float filter[5][5]; 	// Only 25 entries -> No shared memory bank conflict 
			if (TID_x < size_filter && TID_y < size_filter)
			{
				filter[TID_x][TID_y] = d_map.C3_param[BID_x][c][TID_x][TID_y];
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
	float *inputs, float *outputs 
)
{
	// Get index info 
	int BID_x = blockIdx.x; 	// foreach: output image 	~6 or ~16 
	int BID_y = blockIdx.y; 	// foreach: BATCH among curr_step_inputs[BATCH_SIZE] 
	int TID_x = threadIdx.x; 	// foreach: output image row 	~14 or ~5
	int TID_y = threadIdx.y; 	// foreach: output image column 	~14 or ~5

	float acc = 0;
	if (stage == 2)
	{// S2_layer pooling: D_BATCH_SIZE * { Sigmoid([6 @ 28 * 28] + bias[6]) => [6 @ 14 * 14] }
	 	// No need to load C1_bias since it will be cached into L1
		float *input_start = inputs + (BID_y * 6 * 28 * 28) + (BID_x * 28 * 28);
	 	for (int s_row = 0; s_row < 2; s_row++)
	 	{
		 	for (int s_col = 0; s_col < 2; s_col++)
		 	{
				acc += input_start[(28 * (2 * TID_x + s_row)) + (2 * TID_y + s_col)] / 4;
		 	}
	 	}
	 
	 	outputs[(BID_y * 6 * 14 * 14) + (BID_x * 14 * 14) + (TID_x * 14) + TID_y] = sigmoid(acc + d_map.C1_bias[BID_x]);
	}
	else // Desired stage = 4
	{// S4_layer pooling: D_BATCH_SIZE * { Sigmoid([16 @ 10 * 10] + bias[16]) => [16 @ 5 * 5] }
		// No need to load C3_bias since it will be cached into L1 
		float *input_start = inputs + (BID_y * 16 * 10 * 10) + (BID_x * 10 * 10);
		for (int s_row = 0; s_row < 2; s_row++)
		{
			for (int s_col = 0; s_col < 2; s_col++)
			{
				acc += input_start[(10 * (2 * TID_x + s_row)) + (2 * TID_y + s_col)] / 4;
			}
		}

		outputs[(BID_y * 16 * 5 * 5) + (BID_x * 5 * 5) + (TID_x * 5) + TID_y] = sigmoid(acc + d_map.C3_bias[BID_x]); 
	}
	return;
}

__global__ void 	// Fully connecting computation kernel 
fullyConnect_kernel(
	int curr_step, int stage,
	int size_input, int size_output,
	__gpu_map_spill__ *d_map_spill,
	float *inputs, float *outputs 
)
{
	// This layer is pretty much simple matrix multipliction of (ex [120][400] X [400][1] => [120][1] )
	int BID_x = blockIdx.x; 	// I will divide [120][140] into 4 segments, to acquire more blocks for latency hiding 
	int BID_y = blockIdx.y; 	// Unit position in BATCH_SIZE 
	int TID_x = threadIdx.x; 	// Thread ID. threads ~400 or ~120
	if (stage == 5)
	{// F5_layer full connection: D_BATCH_SIZE * { Sigmoid([120 * 400] X Serial[16 @ 5 * 5] + bias[120 * 1]) => [120 * 1] }
		// Load input data into shared memory 
		// Loading F5_param is unnecessary, since elements in F5_param are only for one-shot use 
		__shared__ float prod_elementwise[400];
		__shared__ float input[400];
		if (TID_x < 20)
		{// Take 20 strides to avoid shared memory bank conflict
			for (int i = 0; i < (400 / 20); i++)
			{
				input[(i * 20) + TID_x] = inputs[(BID_y * 400) + (i * 20) + TID_x]; 
			}
		}
		__syncthreads();

		for (int i = 0; i < (120 / 4); i++)
		{
			prod_elementwise[TID_x] = (*d_map_spill).F5_param[((BID_x * (120 / 4)) + i)][TID_x] * input[TID_x];
			__syncthreads();
			if (TID_x == 0)
			{
				float prod_sum = 0;
				for (int j = 0; j < 400; j++)
				{
					prod_sum += prod_elementwise[j];
				}
				outputs[(BID_y * 120) + (BID_x * (120 / 4)) + i] = sigmoid(prod_sum + d_map.F5_bias[(BID_x * (120 / 4) + i)]);
			}
		}
	}
	else // Desired stage = 6
	{// F6_layer full connection: D_BATCH_SIZE * { Sigmoid([84 * 120] X [120 * 1] + bias[84 * 1]) => [84 * 1] }
		// Load input data into shared memory 
		// Loading F6_param is unnecessary, since elements in F6_param are only for one-shot use 
		__shared__ float prod_elementwise[120];
		__shared__ float input[120];
		if (TID_x < 20)
		{// Take 20 strides to avoid shared memory bank conflict 
			for (int i = 0; i < (120 / 20); i++)
			{
				input[(i * 20) + TID_x] = inputs[(BID_y * 120) + (i * 20) + TID_x];
			}
		}
		__syncthreads();

		for (int i = 0; i < (84 / 4); i++)
		{
			prod_elementwise[TID_x] = d_map.F6_param[(BID_x * (120 / 4)) + i][TID_x] * input[TID_x];
			__syncthreads();
			if (TID_x == 0)
			{
				float prod_sum = 0;
				for (int j = 0; j < 120; j++)
				{
					prod_sum += prod_elementwise[j];
				}
				outputs[(BID_y * 84) + (BID_x * (84 / 4)) + i] = sigmoid(prod_sum + d_map.F6_bias[(BID_x * (84 / 4)) + i]);
			}
		}
	}
	return;
}

__global__ void 	// Output layer compuation kernel 
output_kernel(
	int curr_step, int stage,
	int size_input, int size_output,
	__gpu_map_spill__ *d_map_spill,
	float *inputs, float *outputs
)
{
	// OUTPUT_layer: D_BATCH_SIZE * { [10 * 84] X [84 * 1] + [10 * 1] => [10 * 1] }
	// Get index info 
	int BID_y = blockIdx.y; 	// foreach: BATCH among curr_step_inputs[BATCH_SIZE] 
	int TID_x = threadIdx.x; 	// foreach: elements in a row 
	
	// Load data into shared memory 
	__shared__ float OUTPUT_param[10][84];
	if (TID_x < 21)
	{
		for (int i = 0; i < 10; i++)
		{
			for (int j = 0; j < 4; j++)
			{
				OUTPUT_param[i][(j * 21) + TID_x] = d_map.OUTPUT_param[i][(j * 21) + TID_x];
			}
		}
	}
	__syncthreads();	
	__shared__ float input[84];
	if (TID_x < 21)
	{
		for (int i = 0; i < 4; i++)
		{
			input[(i * 21) + TID_x] = inputs[(BID_y * 84) + (i * 21) + TID_x];
		}
	}
	__syncthreads();

	__shared__ float prod_elementwise[84];
	for (int i = 0; i < 10; i++)
	{
		prod_elementwise[TID_x] = OUTPUT_param[i][TID_x] * input[TID_x];
		__syncthreads();
		if (TID_x == 0)
		{
			float prod_sum = 0;
			for (int j = 0; j < 84; j++)
			{
				prod_sum += prod_elementwise[j];
			}
			outputs[(curr_step * D_BATCH_SIZE * 10) + (BID_y * 10) + i] = prod_sum + d_map.OUTPUT_bias[i];
		}
	}
	return;
}

__global__ void 	// Number determination kernel 
numberDetermine_kernel(
	int curr_step, int stage,
	float *inputs, int *outputs
)
{
	// NUMBER_layer: D_NUM_TEST * { ReduceMax[10 * 1] => SINGLE_DIGIT }
	// Get index info 
	int BID_x = blockIdx.x; // 100
	int TID_x = threadIdx.x; // 100

	int index_image = (BID_x * 100) + TID_x;

	float highest_prob = inputs[(index_image * 10) + 0];
	int ans = 0;

	for (int i = 1; i < 10; i++)
	{
		if (inputs[(index_image * 10) + i] > highest_prob)
		{
			highest_prob = inputs[(index_image * 10) + i];
			ans = i;
		}
	}

	outputs[index_image] = ans;
	return;
}

void forward_GPU(float **ptr_test_data, int **ptr_test_label, __map__ *map, int *cnt_correct)
{// Deploy forward computation job on GPU
	float *test_data = *ptr_test_data;
	int *test_label = *ptr_test_label;

	// Acquire memory space in GPU 
	// Prefix "d_" means ADDRESS in device memory 
	// Handlers for device memory manipulation
	int *inferences = malloc(sizeof(int) * NUM_TEST);
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
	cudaMalloc((void **) &d_inferences, sizeof(int) * NUM_TEST);
	cudaMalloc((void **) &d_test_data, sizeof(float) * NUM_TEST * 32 * 32);
	cudaMalloc((void **) &d_map_spill, sizeof(__gpu_map_spill__));
	cudaMalloc((void **) &d_c1_results, sizeof(float) * BATCH_SIZE * 6 * 28 * 28);
	cudaMalloc((void **) &d_s2_results, sizeof(float) * BATCH_SIZE * 6 * 14 * 14);
	cudaMalloc((void **) &d_c3_results, sizeof(float) * BATCH_SIZE * 16 * 10 * 10);
	cudaMalloc((void **) &d_s4_results, sizeof(float) * BATCH_SIZE * 16 * 5 * 5);
	cudaMalloc((void **) &d_f5_results, sizeof(float) * BATCH_SIZE * 120);
	cudaMalloc((void **) &d_f6_results, sizeof(float) * BATCH_SIZE * 84);
	cudaMalloc((void **) &d_output_results, sizeof(float) * NUM_TEST * 10);

	// CUDA memcpy from host to device 
	//cudaMemcpyToSymbol(D_NUM_TEST, &d_NUM_TEST, sizeof(int), 0, cudaMemcpyHostToDevice);
	//cudaMemcpyToSymbol(D_BATCH_SIZE, &batch_size, sizeof(int), 0, cudaMemcpyHostToDevice);
	//cudaMemcpyToSymbol(d_map, tmp_map, sizeof(__gpu_map__), 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(D_NUM_TEST, (void *) &d_NUM_TEST, sizeof(int));
	cudaMemcpyToSymbol(D_BATCH_SIZE, (void *) &batch_size, sizeof(int));
	cudaMemcpyToSymbol(d_map, (void *) tmp_map, sizeof(__gpu_map__));
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
		// 0. Convolution layer C1 
		block.x = 6;
		block.y = BATCH_SIZE;
		block.z = 1;
		thread.x = 28;
		thread.y = 28;
		thread.z = 1;
		convolution_kernel<<<block, thread>>>(step, 1, 6, 1, 32, 32, 5, d_test_data, d_c1_results);

		// 1. Pooling layer S2 
		block.x = 6;
		block.y = BATCH_SIZE;
		block.z = 1;
		thread.x = 14;
		thread.y = 14;
		thread.z = 1;
		pooling_kernel<<<block, thread>>>(step, 2, 6, 28, 28, d_c1_results, d_s2_results);
		
		// 2. Convolution layer C3 
		block.x = 16;
		block.y = BATCH_SIZE;
		block.z = 1;
		thread.x = 10;
		thread.y = 10;
		thread.z = 1;
		convolution_kernel<<<block, thread>>>(step, 3, 16, 6, 14, 14, 5, d_s2_results, d_c3_results);

		// 3. Pooling layer S4 
		block.x = 16;
		block.y = BATCH_SIZE;
		block.z = 1;
		thread.x = 5;
		thread.y = 5;
		thread.z = 1;
		pooling_kernel<<<block, thread>>>(step, 4, 16, 10, 10, d_c3_results, d_s4_results);

		// 4. Fully connected layer F5 
		block.x = 4;
		block.y = BATCH_SIZE;
		block.z = 1;
		thread.x = 400;
		thread.y = 1;
		thread.z = 1;
		fullyConnect_kernel<<<block, thread>>>(step, 5, 400, 120, d_map_spill, d_s4_results, d_f5_results);

		// 5. Fully connected layer F6 
		block.x = 4;
		block.y = BATCH_SIZE;
		block.z = 1;
		thread.x = 120;
		thread.y = 1;
		thread.z = 1;
		fullyConnect_kernel<<<block, kernel>>>(step, 6, 120, 84, d_map_spill, d_f5_results, d_f6_results);

		// 6. Output layer OUTPUT 
		block.x = 1;
		block.y = BATCH_SIZE;
		block.z = 1;
		thread.x = 84;
		thread.y = 1;
		thread.z = 1;
		output_kernel<<<block, kernel>>>(step, 7, 84, 10, d_map_spill, d_f6_results, d_output_results);
	}

	// 7. Determine numbers
	block.x = 100;
	block.y = 1;
	block.z = 1;
	thread.x = 100;
	thread.y = 1;
	thread.z = 1;
	numberDetermine_kernel<<<block, thread>>>(8, 8, d_output_results, d_inferences);

	// 8. Copy inference answers to Host 
	cudaMemcpy(inferences, d_inferences, sizeof(int) * NUM_TEST, cudaMemcpyDeviceToHost);

	// 9. Scoring
	for (int i = 0; i < NUM_TEST; i++)
	{
		if (inference[i] == test_label[i])
		{
			(*cnt_correct)++;
		}
	}

	// WARNING: FREE 0
	free(inferences);
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

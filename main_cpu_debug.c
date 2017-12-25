#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include "parser.h"

#define DEBUG 1

const char *PATH_TEST_DATA = "./mnist/t10k-images-idx3-ubyte";
const char *PATH_TEST_LABEL= "./mnist/t10k-labels-idx1-ubyte";

float sigmoid(float x)
{
	return (exp(x) / (1 + exp(x)));
}

void prompt(void)
{
		printf("Press enter to continue: \n");
		char asdf;
		fgets(&asdf, 100, stdin);
		return;
}

void convolution(int num_input, int num_output, int height_input, int width_input, int size_filter, float *pre_filters, float *pre_inputs, float *pre_outputs)
{// Performs convolution computation 
	/*
	 * num_input: number of input images 
	 * num_output: number of output images 
	 * height_input: height of each input image
	 * width_input: width of each input image
	 * size_filter: height (and width) of each filter map 
	 * inputs: input images 
	 * filters: convolution filters
	 * outputs: output images
	 */

	int height_output = height_input - size_filter + 1;
	int width_output = width_input - size_filter + 1;

	float inputs[num_input][height_input][width_input];
	float outputs[num_output][height_output][width_output];
	float filters[num_output][num_input][size_filter][size_filter];

	memcpy(inputs, pre_inputs, sizeof(float) * num_input * height_input * width_input);
	memcpy(filters, pre_filters, sizeof(float) * num_output * num_input * size_filter * size_filter);

	for (int o = 0; o < num_output; o++)
	{// For each output image 
		for (int h = 0; h < height_output; h++)
		{// For each row
			for (int w = 0; w < width_output; w++)
			{// For each element
				outputs[o][h][w] = 0;
				for (int i = 0; i < num_input; i++)
				{// For each input
					for (int f_h = 0; f_h < size_filter; f_h++)
					{// For each filter row 
						for (int f_w = 0; f_w < size_filter; f_w++)
						{// For each filter element
							outputs[o][h][w] += (inputs[i][h + f_h][w + f_w] * filters[o][i][f_h][f_w]);
						}
					}
				}
			}
		}
	}

	memcpy(pre_outputs, outputs, sizeof(float) * num_output * height_output * width_output);

#ifdef DEBUG 
	printf("Printing convolution results...\n");
	printf("\n --------------------------------- \n");
	for (int i = 0; i < num_output; i++)
	{
		for (int j = 0; j < height_output; j++)
		{
			for (int k = 0; k < width_output; k++)
			{
				printf("%f ", outputs[i][j][k]);
			}
			printf("\n");
		}
		printf("\n ------------------------------------ \n");
	}
	printf("\n --------------------------------- \n");
#endif 

	return;
}

void pooling(int num_output, int height_input, int width_input, float *pre_inputs, float *pre_outputs, float *bias)
{// Consolidate matrix
	/*
	 * num_output: number of input images
	 * height_input: height of each input image
	 * width_input: width of each input image
	 * K: fixed value = 2 in the LeNet - 5 
	 * inputs: input images
	 * outputs: output images
	 */

	int height_output = height_input / 2;
	int width_output = width_input / 2;

	float inputs[num_output][height_input][width_input];
	float outputs[num_output][height_output][width_output];

	memcpy(inputs, pre_inputs, sizeof(float) * num_output * height_input * width_input);

	for (int o = 0; o < num_output; o++)
	{// For each output image
		for (int h = 0; h < (height_input / 2); h++)
		{// Till half of input height reached
			for (int w = 0; w < (width_input / 2); w++)
			{// Till half of intput width reached 
				outputs[o][h][w] = 0;
				for (int s_h = 0; s_h < 2; s_h++)
				{// 2x2 window, row 
					for (int s_w = 0; s_w < 2; s_w++)
					{// 2x2 window, column
						outputs[o][h][w] += inputs[o][2 * h + s_h][2 * w + s_w] / (2 * 2);
					}
				}

				// Add bias term
				outputs[o][h][w] = sigmoid(outputs[o][h][w] + bias[o]);
			}
		}
	}

	memcpy(pre_outputs, outputs, sizeof(float) * num_output * height_output * width_output);

#ifdef DEBUG 	
	printf("Printing pooling results...\n");
	printf("\n --------------------------------- \n");
	for (int i = 0; i < num_output; i++)
	{
		for (int j = 0; j < height_input; j++)
		{
			for (int k = 0; k < width_input; k++)
			{
				printf("%f ", outputs[i][j][k]);
			}
			printf("\n");
		}
		printf("\n ------------------------------------ \n");
	}
	printf("\n --------------------------------- \n");
#endif 

	return;
}

void fullyConnect(int size_input, int size_output, float *input, float *output, float *pre_fc_params, float *bias)
{// Flatten all features

	float fc_params[size_output][size_input];
	memcpy(fc_params, pre_fc_params, sizeof(float) * size_output * size_input);

	for (int o = 0; o < size_output; o++)
	{// For every output element 
		output[o] = 0;
		for (int i = 0; i < size_input; i++)
		{
			output[o] += input[i] * fc_params[o][i];
		}
		output[o] = sigmoid(output[o] + bias[o]);
	}

#ifdef DEBUG 	
	printf("Printing fullyConnect results...\n");
	printf("\n --------------------------------- \n");
	for (int i = 0; i < size_output; i++)
	{
		printf("%f ", output[i]);
	}
	printf("\n --------------------------------- \n");
#endif 

	return;
}

void output(int size_input, int size_output, float *input, float *output, float *pre_output_params, float *bias)
{// Final output layer

	float output_params[size_output][size_input];
	memcpy(output_params, pre_output_params, sizeof(float) * size_output * size_input);

	for (int o = 0; o < size_output; o++)
	{// For every output element 
		output[o] = 0;
		for (int i = 0; i < size_input; i++)
		{
			output[o] += input[i] * output_params[o][i];
		}
		output[o] += bias[o];
	}

#ifdef DEBUG 	
	printf("Printing output results...\n");
	printf("\n --------------------------------- \n");
	for (int i = 0; i < size_output; i++)
	{
		printf("%f ", output[i]);
	}
	printf("\n --------------------------------- \n");
#endif 

	return;
}

void check(float *input, int label, int *cnt)
{
	float highestProb = input[0];
	int ans = 0;
	for (int i = 1; i < 10; i++)
	{
		if (highestProb < input[i])
		{
			highestProb = input[i];
			ans = i;
		}
	}

	if (ans == label)
	{
		(*cnt)++;
	}

#ifdef DEBUG 
	printf("PREDICTION: %d vs ANSWER: %d\n", ans, label);
	printf("\n");
#endif

	return;
}

int main(int argc, char *argv[])
{
	printf("Started...\n");
	// Initialize map variable
	__map__ map;
	printf("Loading weights...\n");
	load_weights(&map);	

	// WARNING: MALLOC 
	// Storage for intermediate variables 
	float *c1_result = malloc(sizeof(float) * 6 * 28 * 28);
	float *s2_result = malloc(sizeof(float) * 6 * 14 * 14);
	float *c3_result = malloc(sizeof(float) * 16 * 10 * 10);
	float *s4_result = malloc(sizeof(float) * 16 * 5 * 5);
	float *f5_result = malloc(sizeof(float) * 120);
	float *f6_result = malloc(sizeof(float) * 84);
	float *output_result = malloc(sizeof(float) * 10);

	// WARNING: MALLOC 
	float *test_data = malloc(sizeof(float) * NUM_TEST * 32 * 32);
	int *test_label = malloc(sizeof(int) * NUM_TEST);

	// LOAD test data and labels
	float *data = malloc(sizeof(float) * NUM_TEST * 32 * 32);
	int *label = malloc(sizeof(int) * NUM_TEST);

	printf("Malloc completed\n");

	read_data(PATH_TEST_DATA, data);
	read_label(PATH_TEST_LABEL, label);

	printf("Data read completed\n");

	memcpy(test_data, data, sizeof(float) * NUM_TEST * 32 * 32);
	memcpy(test_label, label, sizeof(int) * NUM_TEST);

	free(data);
	free(label);

	printf("Data load completed\n");

	int cnt = 0;

#ifdef DEBUG 
	printf("Printing out test_label\n");
	for (int i = 0; i < NUM_TEST; i++)
	{
		printf("%d\n", test_label[i]);
	}
	printf("Press enter to continue: \n");
	char asdfs;
	fgets(&asdfs, 100, stdin);
#endif

#ifndef DEBUG 
	clock_t start = clock(), diff;

#endif 

	for (int i = 0; i < NUM_TEST; i++)
	{
#ifdef DEBUG 
		printf("\nIndex %d: \n", i);
#endif 
		// C1 convolution 
		convolution(1, 6, 32, 32, 5, (float *) map.C1_param, &test_data[i * 32 * 32], c1_result);
		// S2 pooling 
		pooling(6, 28, 28, c1_result, s2_result, map.C1_bias);
		// C3 convolution 
		convolution(6, 16, 14, 14, 5, (float *) map.C3_param, s2_result, c3_result);
		// S4 pooling 
		pooling(16, 10, 10, c3_result, s4_result, map.C3_bias);
		// F5 full connection
		fullyConnect(400, 120, s4_result, f5_result, (float *) map.F5_param, map.F5_bias);
		// F6 full connection 
		fullyConnect(120, 84, f5_result, f6_result, (float *) map.F6_param, map.F6_bias);
		// Output layer 
		output(84, 10, f6_result, output_result, (float *) map.OUTPUT_param, map.OUTPUT_bias);
		// Check 
		check(output_result, test_label[i], &cnt);
#ifdef DEBUG 
		printf("\n Actual Image: \n");
		printMNIST(test_data + (i * 32 * 32), test_label[i]);
		printf("Press enter to continue: \n");
		char asdf;
		fgets(&asdf, 100, stdin);
#endif 
	}

#ifndef DEBUG 
	diff = clock() - start;
	int msec = diff * 1000 / CLOCKS_PER_SEC;
	printf("Elapsed time: %d seconds %d milliseconds", msec / 1000, msec % 1000);
#endif 

	float accuracy = ((float) cnt) / ((float) NUM_TEST);
	printf("Prediction accuracy: %f%%\n", accuracy * 100);

	free(c1_result);
	free(s2_result);
	free(c3_result);
	free(s4_result);
	free(f5_result);
	free(f6_result);
	free(output_result);
	free(test_data);
	free(test_label);

	return 0;
}

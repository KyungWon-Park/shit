#include <stdio.h>

void convLayer_forward(int M, int C, int H_in, int W_in, int k, float *W, float *X, float *Y)
{
	/*
	 * C: number of input feature maps
	 * M: number of output feature maps
	 * H_in: height of each input image
	 * W_in: width of each input map image
	 * K: height (and width) of each filter bank
	 * X: input feature maps
	 * W: convolution filters
	 * Y: output feature maps
	 */
	
	return;
}

void poolingLayer_forward(int M, int H_in, int W_in, float *Y, float *S)
{
	/*
	 * M: number of output feature maps
	 * H_in: height of each input image
	 * W_in: width of each input map image
	 * K: fixed value = 2 in the LeNet - 5
	 * Y: input feature maps 
	 * S: output feature maps
	 */

	return;
}

int main(int argc, char *argv[])
{
	return 0;
}

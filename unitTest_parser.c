#include "parser.h"

const char *PATH_TEST_DATA = "./mnist/t10k-images-idx3-ubyte";
const char *PATH_TEST_LABEL= "./mnist/t10k-labels-idx1-ubyte";

int main(int argc, char *argv[])
{
	float *data = malloc(sizeof(float) * NUM_TEST * 32 * 32 * 10);
	assert(data != NULL && "Failed to assign data\n");
	int *labels = malloc(sizeof(int) * NUM_TEST * 10);
	assert(labels != NULL && "Failed to assign labels\n");

	read_data(PATH_TEST_DATA, data);
	read_label(PATH_TEST_LABEL, labels);

	while (1)
	{
		printf("\nEnter index of image to see: ");
		int n;
		scanf("%d", &n);
		printf("\n [ Number: %d ] \n\n", labels[n]);
		printMNIST(data + (n * 32 * 32), labels[n]);	
	}

	free(data);
	free(labels);
	return 0;
}

#include "parser.h"

void load_weights(__map__ *ptr_map)
{
	printf("Loading weight parameters...\n");
	FILE *fp;
	
	// LOAD C1 BIAS
	{
		printf("LOADING C1_BIAS\n");
		fp = fopen("./weights/C1_bias.txt", "r");
		fscanf(fp, "%e %e %e %e %e %e", &(*ptr_map).C1_bias[0], &(*ptr_map).C1_bias[1], &(*ptr_map).C1_bias[2], &(*ptr_map).C1_bias[3], &(*ptr_map).C1_bias[4], &(*ptr_map).C1_bias[5]);
		fclose(fp);
	}

	// LOAD C1 PARAM
	{
		printf("LOADING C1_PARAM\n");
		fp = fopen("./weights/C1_param.txt", "r");
		for (int i = 0; i < 6; i++)
		{
			for (int j = 0; j < 5; j++)
			{
				float tmp[5];
				fscanf(fp, "%e %e %e %e %e", &tmp[0], &tmp[1], &tmp[2], &tmp[3], &tmp[4]);
				for (int k = 0; k < 5; k++)
				{
					(*ptr_map).C1_param[i][0][j][k] = tmp[k];
				}
			}
		}
		fclose(fp);
	}

	// LOAD C3 BIAS 
	{
		printf("LOADING C3_BIAS\n");
		fp = fopen("./weights/C3_bias.txt", "r");
		float tmp[16];
		for (int i = 0; i < 16; i++)
		{
			fscanf(fp, "%e", &(*ptr_map).C3_bias[i]);
		}
		fclose(fp);
	}

	// LOAD C3 PARAM
	{
		printf("LOADING C1_PARAM\n");
		fp = fopen("./weights/C3_param.txt", "r");
		for (int i = 0; i < 16; i++)
		{
			for (int j = 0; j < 6; j++)
			{
				for (int k = 0; k < 5; k++)
				{
					float tmp[5];
					fscanf(fp, "%e %e %e %e %e", &tmp[0], &tmp[1], &tmp[2], &tmp[3], &tmp[4]);
					for (int l = 0; l < 5; l++)
					{
						(*ptr_map).C3_param[i][j][k][l] = tmp[l];
					}
				}
			}
		}
		fclose(fp);
	}

	// LOAD F5 BIAS 
	{
		printf("LOADING F5_BIAS\n");
		fp = fopen("./weights/F5_bias.txt", "r");
		for (int i = 0; i < 120; i++)
		{
			fscanf(fp, "%e", &(*ptr_map).F5_bias[i]);
		}
		fclose(fp);
	}

	// LOAD F5 PARAM 
	{
		printf("LOADING F5_PARAM\n");
		fp = fopen("./weights/F5_param.txt", "r");
		for (int i = 0; i < 120; i++)
		{
			for (int j = 0; j < 400; j++)
			{
				fscanf(fp, "%e", &(*ptr_map).F5_param[i][j]);
			}
		}
		fclose(fp);
	}

	// LOAD F6 BIAS 
	{
		printf("LOADING F6_BIAS\n");
		fp = fopen("./weights/F6_bias.txt", "r");
		for (int i = 0; i < 84; i++)
		{
			fscanf(fp, "%e", &(*ptr_map).F6_bias[i]);
		}
		fclose(fp);
	}

	// LOAD F6 PARAM 
	{
		printf("LOADING F6_PARAM\n");
		fp = fopen("./weights/F6_params.txt", "r");
		for (int i = 0; i < 84; i++)
		{
			for (int j = 0; j < 120; j++)
			{
				fscanf(fp, "%e", &(*ptr_map).F6_param[i][j]);
			}
		}
		fclose(fp);
	}

	// LOAD OUTPUT BIAS 
	{
		printf("LOADING OUTPUT_BIAS\n");
		fp = fopen("./weights/output_bias.txt", "r");
		for (int i = 0; i < 10; i++)
		{
			fscanf(fp, "%e", &(*ptr_map).OUTPUT_bias[i]);
		}
		fclose(fp);
	}

	// LOAD OUTPUT PARAM 
	{
		printf("LOADING OUTPUT_PARAM\n");
		fp = fopen("./weights/output_params.txt", "r");
		for (int i = 0; i < 10; i++)
		{
			for (int j = 0; j < 84; j++)
			{
				fscanf(fp, "%e", &(*ptr_map).OUTPUT_param[i][j]);
			}
		}
		fclose(fp);
	}

	return;
}

int reverse_int32(int i)
{
	unsigned char byte1, byte2, byte3, byte4;
	byte1 = i & MAXBYTE; 
	byte2 = (i >> 8) & MAXBYTE; 
	byte3 = (i >> 16) & MAXBYTE; 
	byte4 = (i >> 24) & MAXBYTE; 

	return ((int) byte1 << 24) + ((int) byte2 << 16) + ((int) byte3 << 8) + ((int) byte4);
}

void read_data(const char* datapath, float *data)
{
	printf("Starting to load MNIST data...\n");
	// Open file 
	FILE *fp = fopen(datapath, "r");
	assert(fp != NULL && "Failed to open MNIST dataset!\n");

	// Read the header information 
	int magic_number = 0;
	int number_of_images = 0;
	int n_rows = 0;
	int n_cols = 0;

	fread((char *) &magic_number, sizeof(magic_number), 1, fp);
	magic_number = reverse_int32(magic_number);
	printf("Magic number: %d\n", magic_number);

	fread((char *) &number_of_images, sizeof(number_of_images), 1, fp);
	number_of_images = reverse_int32(number_of_images);
	printf("Number of images: %d\n", number_of_images);

	fread((char *) &n_rows, sizeof(n_rows), 1, fp);
	n_rows = reverse_int32(n_rows);

	fread((char *) &n_cols, sizeof(n_cols), 1, fp);
	n_cols = reverse_int32(n_cols);

	printf("[ Size of row: %d | Size of column: %d ]\n", n_rows, n_cols);

	// Read actual MNIST data set (uint8 -> float)
	for (int i = 0; i < number_of_images; i++)
	{
#ifdef DEBUG 
		printf("LOADING: %dth \n", i);
#endif
		for (int r = 0; r < n_rows; r++)
		{
			for (int c = 0; c < n_cols; c++)
			{
				unsigned char tmp = 0;
				if ((r >= 2 || r <= 29) && (c >= 2 || c <= 29))
				{
					fread((char *) &tmp, sizeof(tmp), 1, fp);
				}
				data[(i * 32 * 32) + (n_rows * r) + c] = (float) tmp * (float) MNIST_SCALE_FACTOR;
			}
#ifdef DEBUG 
			printf("LOADED: %dth \n", i);
#endif
		}
	}
	fclose(fp);

	printf("Finished loading MNIST data set\n\n");
	return; 
}

void read_label(const char *labelPath, int *labels)
{
	printf("Starting to load MNIST label data...\n");
	int number_of_labels = 0;

	// Open file 
	FILE *fp = fopen(labelPath, "r");
	assert(fp != NULL && "Failed to open MNIST label file!\n");

	int magic_number = 0;

	// Read label information 
	fread((char *) &magic_number, sizeof(magic_number), 1, fp);
	magic_number = reverse_int32(magic_number);
	printf("Magic number: %d\n", magic_number);

	fread((char *) &number_of_labels, sizeof(number_of_labels), 1, fp);
	number_of_labels = reverse_int32(number_of_labels);
	printf("Number of labels: %d\n", number_of_labels);

	// Load label data 
	for (int i = 0; i < number_of_labels; i++)
	{
		unsigned char tmp = 0;
		fread((char *) &tmp, sizeof(tmp), 1, fp);
		labels[i] = (int) tmp;
	}

	fclose(fp);
	printf("Finished loading label data\n");
	return;
}

void printMNIST(float *data, int label)
{
	printf("LABEL answer: %d\n", label);

	for (int r = 0; r < 32; r++)
	{
		for (int c = 0; c < 32; c++)
		{
			if (data[r * RAW_DIM + c] > 0.5f)
			{
				printf("■");
			}
			else 
			{
				printf("□");
			}
		}
		printf("\n");
	}

	return;
}

#include <stdio.h>
#include "parser.h"

void load_weights(__map__ *ptr_map)
{
	FILE *fp;
	
	// LOAD C1 BIAS
	{
		printf("LOADING C1_BIAS\n");
		fp = fopen("./weights/C1_bias.txt", "r");
		float tmp[6];
		fscanf(fp, "%e %e %e %e %e %e", &tmp[0], &tmp[1], &tmp[2], &tmp[3], &tmp[4], &tmp[5]);
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

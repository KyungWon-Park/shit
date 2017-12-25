#include <stdio.h>

int main(void)
{
	FILE *fp = fopen("./mnist/t10k-images-idx3-ubyte", "r");
	float tmp;
	for (int i = 0; i < 32; i++)
	{
		for (int j = 0; j < 32; j++)
		{
			fread(&tmp, 4, 1, fp);
			printf("%f ", tmp);
		}
		printf("\n");
	}
	return 0;
}

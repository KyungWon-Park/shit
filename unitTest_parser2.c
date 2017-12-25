#include "parser.h"

int main(void)
{
	__map__ map;
	printf("Started to load weight parameters...\n");
	load_weights(&map);
	printf("Weight parameter loading completed\n");

	printf("Enter number to see parameter: (1: C1 / 2: C3 / 3: F5 / 4: F6 / 5: OUTPUT");
	int btn;
	scanf("%d", &btn);

	switch (btn)
	{
		case 1:
			printf("PRINTING C1_param\n\n");
			for (int i = 0; i < 6; i++)
			{
				for (int j = 0; j < 1; j++)
				{
					for (int k = 0; k < 5; k++)
					{
						for (int l = 0; l < 5; l++)
						{
							printf("%f ", map.C1_param[i][j][k][l]);
						}
						printf("\n");
					}
				}
				printf("\n ------------------------------- \n");
			}
			break;
		case 2:
		case 3:
		case 4:
		case 5:
		default:
			printf("Not ready yet man\n");
			break;
	}

	return 0;
}

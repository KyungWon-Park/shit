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
			printf("PRINTING C3_param\n\n");
			for (int i = 0; i < 16; i++)
			{
				for (int j = 0; j < 6; j++)
				{
					for (int k = 0; k < 5; k++)
					{
						for (int l = 0; l < 5; l++)
						{
							printf("%f ", map.C3_param[i][j][k][l]);
						}
						printf("\n"); 
					}
					printf("\n -------------------------------- \n");
				}
				printf("\n ---------------------------------- \n");
			}
			break;
		case 3:
			printf("PRINITNG F5_param\n\n");
			for (int i = 0; i < 120; i++)
			{
				for (int j = 0; j < 400; j++)
				{
					printf("%f ", map.F5_param[i][j]);
				}
				printf("\n");
			}
			break;
		case 4:
			printf("PRINITNG F6_param\n\n");
			for (int i = 0; i < 84; i++)
			{
				for (int j = 0; j < 120; j++)
				{
					printf("%f ", map.F6_param[i][j]);
				}
				printf("\n");
			}
			break; 
		case 5:
			printf("PRINTING OUTPUT_param\n\n");
			for (int i = 0; i < 10; i++)
			{
				for (int j = 0; j < 84; j++)
				{
					printf("%f ", map.OUTPUT_param[i][j]);
				}
				printf("\n");
			}
			break;
		default:
			printf("Not ready yet man\n");
			break;
	}

	return 0;
}

typedef struct 
{
	// float C1_param[6][1][5][5];
	float C1_param[6 * 1 * 5 * 5];
	// float C1_bias[6];
	float C1_bias[6];
	// float C3_param[16][6][5][5];
	float C3_param[16 * 6 * 5 * 5];
	// float C3_bias[16];
	float C3_bias[16];
	// float F5_param[120][400];
	float F5_param[120 * 400];
	// float F5_bias[120];
	float F5_bias[120];
	// float F6_param[84][120];
	float F6_param[84 * 120];
	// float F6_bias[84];
	float F6_bias[84];
	// float OUTPUT_param[10][84];
	float OUTPUT_param[10 * 84];
	// float OUTPUT_bias[10];
	float OUTPUT_bias[10];
} __map__; 

/* 
	IC621 2017 MNIST reader 
	written by Jinwook Kim (Dec. 05, 2017)
*/
#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <fstream>
using namespace std;

#define NUM_TRAIN 60000
#define NUM_TEST  10000
#define RAW_DIM   28
#define RAW_PIXELS_PER_IMG 784 			// 28x28, single channel image
#define MNIST_SCALE_FACTOR 0.00390625	// 1/255
#define MAXBYTE 255

const char* PATH_TRAIN_DATA = "/home/ic621/mnist/train-images-idx3-ubyte";
const char* PATH_TRAIN_LABEL = "/home/ic621/mnist/train-labels-idx1-ubyte";
const char* PATH_TEST_DATA = "/home/ic621/mnist/t10k-images-idx3-ubyte";
const char* PATH_TEST_LABEL= "/home/ic621/mnist/t10k-labels-idx1-ubyte";

/***** Function declarations ***************************/
void printMNIST(float* data, int label);
void read_data(const char* datapath, float** data);
void read_label(const char* labelPath, int* label);

/***** Main function ***********************************/
int main(){
	
	float** Xtrain;
	float** Xtest;
	int* Ytrain; 
	int* Ytest;
	int checkLabel;
	// allocate memory for MNIST data/labels
	Xtrain = (float**)malloc(sizeof(float*)*NUM_TRAIN);
	for(int i=0; i<NUM_TRAIN; i++){ Xtrain[i] = (float*)malloc(sizeof(float)*RAW_PIXELS_PER_IMG); }
	Ytrain = (int*)malloc(sizeof(int)*NUM_TRAIN);
	
	Xtest = (float**)malloc(sizeof(float*)*NUM_TEST);
	for(int i=0; i<NUM_TEST; i++){ Xtest[i] = (float*)malloc(sizeof(float)*RAW_PIXELS_PER_IMG); }
	Ytest = (int*)malloc(sizeof(int)*NUM_TEST);
	
	// load data
	read_data (PATH_TRAIN_DATA, Xtrain);
	read_label(PATH_TRAIN_LABEL, Ytrain);
	
	read_data (PATH_TEST_DATA, Xtest);
	read_label(PATH_TEST_LABEL, Ytest);
	
	// check data (optional)
	checkLabel = 0;
	printMNIST(Xtrain[checkLabel], Ytrain[checkLabel]);
	printMNIST(Xtest[checkLabel], Ytest[checkLabel]);
	checkLabel = 59999;
	printMNIST(Xtrain[checkLabel], Ytrain[checkLabel]);
	checkLabel = 9999;
	printMNIST(Xtest[checkLabel], Ytest[checkLabel]);
	
	// deallocate memory
	for(int i=0; i<NUM_TRAIN; i++){ free(Xtrain[i]) ; Xtrain[i]=NULL;}
	free(Xtrain); Xtrain = NULL;
	for(int i=0; i<NUM_TEST; i++){ free(Xtest[i]) ; Xtest[i]=NULL;}
	free(Xtest); Xtest = NULL;
	free(Ytrain); Ytrain = NULL;
	free(Ytest); Ytest = NULL;
}

/***** Function definitions ***************************/

int reverse_int32 (int i){
	unsigned char byte1, byte2, byte3, byte4;
	byte1 = i&MAXBYTE;
	byte2 = (i>>8)&MAXBYTE;
	byte3 = (i>>16)&MAXBYTE;
	byte4 = (i>>24)&MAXBYTE;
	return ( (int)byte1<<24 ) + ( (int)byte2<<16 ) + ( (int)byte3<<8 ) + (int)byte4;
}
/*
	Read [number_of_images]x28x28 MNIST data from {datapath}
	Store data into the given float array 
*/
void read_data(const char* datapath, float** data){

	ifstream infile (datapath, ios::binary);
	if (!infile.is_open())
	{
		printf("FAILED TO OPEN FILE: %s\n", datapath);
		return ;
	}
	cout << "== Input test image file: " << datapath << endl;
	// read the header information
	int magic_number=0;
	int number_of_images=0;
	int n_rows = 0;
	int n_cols = 0;
	infile.read((char*)&magic_number, sizeof(magic_number));
	magic_number = reverse_int32(magic_number);
	cout << "magic number: " << magic_number << endl;

	infile.read((char*)&number_of_images, sizeof(number_of_images));
	number_of_images = reverse_int32(number_of_images);
	cout << "number of images: " << number_of_images << endl;

	infile.read((char*)&n_rows, sizeof(n_rows));
	n_rows = reverse_int32(n_rows);

	infile.read((char*)&n_cols, sizeof(n_cols));
	n_cols = reverse_int32(n_cols);
	cout << "size of row = " << n_rows << ", size of cols = " << n_cols << endl;
	
	// Read actual data (uint8 -> float)
	for(int i=0; i<number_of_images; ++i)
	{
		data[i][0] = 1;
		for(int r=0; r<n_rows; ++r)
		{
			for(int c=0; c<n_cols; ++c)
			{
				unsigned char temp = 0;
				infile.read((char*)&temp, sizeof(temp));
				data[i][(n_rows*r)+c] = (float)temp * (float)MNIST_SCALE_FACTOR;
			}
		}
	}
	infile.close();
	cout << "Done. [data: "<<datapath <<"] [count: " << number_of_images <<"]"<<endl;
}


void read_label(const char* labelPath, int* labels){
	//char input_file_label[] = "./t10k-labels-idx1-ubyte";
	int number_of_labels = 0;

	ifstream infile(labelPath, ios::binary);
	if (!infile.is_open())
	{
			printf("FAILED TO OPEN FILE: %s\n", labelPath);
			return;
	}
	cout << "== Input test label file: " << labelPath << endl;

	int magic_number=0;
	// read the label information
	infile.read((char*)&magic_number, sizeof(magic_number));
		magic_number = reverse_int32(magic_number);
	cout << "magic number: " << magic_number << endl;

	infile.read((char*)&number_of_labels, sizeof(number_of_labels));
		number_of_labels = reverse_int32(number_of_labels);
	cout << "number of labels: " << number_of_labels << endl;

	
	for(int i=0; i<number_of_labels; ++i)
	{
		unsigned char temp = 0;
		infile.read((char*)&temp, sizeof(temp));
		labels[i] = (int)temp;

	}
	infile.close();
	cout << "Done. [data: "<<labelPath <<"] [count: " << number_of_labels<<"] "<<endl;
}

void printMNIST(float* data, int label){
	cout << "Check data for label " << label << endl;
	for(int r=0; r<RAW_DIM; r++){
		for (int c=0; c<RAW_DIM; c++){
			if (data[r*RAW_DIM+c] > 0.5f){
				cout << "■";
			}
			else {
				cout << "□";
			}
		}
		cout << endl;
	}
}
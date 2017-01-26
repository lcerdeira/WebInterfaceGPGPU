/*Author : H.M.Gamaarachchi
Generate random floating point matrix using CUDA */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda.h>
#include <curand.h>
#include "myhelpers.h"

int main(int argc,char *argv[]){

	//check argc
	if (argc<5){
		fprintf(stderr,"Please enter all arguments eg : ./random rows cols seed file.txt\n");
		exit(EXIT_FAILURE);
	}
	
	//var declaration
	int width=atoi(argv[2]);
	int height=atoi(argv[1]);
	int seed=atoi(argv[3]);
	curandGenerator_t generator;
	float *data, *dev_data;
	
	//mem allocation
	data=(float *)malloc(sizeof(float)*width*height);
	isMemoryFull(data);
	checkCudaError(cudaMalloc((void **)&dev_data,sizeof(float)*width*height));
	
	
	//generate random numbers
	curandCreateGenerator(&generator,CURAND_RNG_PSEUDO_DEFAULT);
	curandSetPseudoRandomGeneratorSeed(generator,seed);
	checkCudaError(cudaGetLastError());
	
	//Time
	cudaEvent_t start,stop;
	float elapsedtime;
	cudaEventCreate(&start);
	cudaEventRecord(start,0);	
	
	curandGenerateUniform(generator,dev_data,width*height);
	checkCudaError(cudaGetLastError());
	
	//time
	cudaEventCreate(&stop);
	cudaEventRecord(stop,0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedtime,start,stop);
	printf("Time spent for calculation in CUDA : %.10f s\n",elapsedtime/(float)1000);	
	
	//memcopy
	checkCudaError(cudaMemcpy(data,dev_data,sizeof(float)*width*height,cudaMemcpyDeviceToHost));
	
	//writing to file
	FILE *fp=fopen(argv[4],"w");
	isFileOK(fp);
	int i,j;
	for (i=0;i<height;i++){
		for (j=0;j<width;j++){
			fprintf(fp,"%f ",data[i*width+j]*10);
		}
		fprintf(fp,"\n");
	}	
	
	//free
	cudaFree(dev_data);
	curandDestroyGenerator(generator);
	free(data);

	return 0;
}
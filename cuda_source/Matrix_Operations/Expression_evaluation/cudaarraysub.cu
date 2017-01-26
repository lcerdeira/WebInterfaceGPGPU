/*
Author : H.M.Gamaarachchi
C file for adding of 2 matrices
*/

/*#include <stdio.h>
#include <string.h>
#include <assert.h>
#include <stdlib.h>
#include "myhelpers.h"

#define BLOCK 16*/


//adding kernel
__global__ void cuda_sub(float *dev_c,float *dev_a,float *dev_b,int width,int height){
	int x = blockDim.x * blockIdx.x + threadIdx.x;
	int y = blockDim.y * blockIdx.y + threadIdx.y;
	if (x<width && y<height){
		int n = y * width + x;
		dev_c[n]=dev_a[n]-dev_b[n];
	}
}

//addition abstraction
cudaError_t substract(float *c,float *a,float *b,int width,int height){
	float *dev_a=0;
	float *dev_b=0;
	float *dev_c=0;
	cudaError_t cudastatus;

	//memory allocation
	cudastatus=cudaMalloc((void**)&dev_a,width*height*sizeof(float));
	if (cudastatus!=cudaSuccess)
		return cudastatus;
	cudastatus=cudaMalloc((void**)&dev_b,width*height*sizeof(float));
	if (cudastatus!=cudaSuccess)
		return cudastatus;
	cudastatus=cudaMalloc((void**)&dev_c,width*height*sizeof(float));
	if (cudastatus!=cudaSuccess)
		return cudastatus;
	
	//copying
	cudastatus=cudaMemcpy(dev_a,a,sizeof(float)*width*height,cudaMemcpyHostToDevice);
	if(cudastatus!=cudaSuccess)
		return cudastatus;
	cudastatus=cudaMemcpy(dev_b,b,sizeof(float)*width*height,cudaMemcpyHostToDevice);
	if(cudastatus!=cudaSuccess) 
		return cudastatus;
	cudastatus=cudaMemcpy(dev_c,c,sizeof(float)*width*height,cudaMemcpyHostToDevice);
	if(cudastatus!=cudaSuccess)
		return cudastatus;
	
	dim3 grid(ceil(width/(float)BLOCK),ceil(height/(float)BLOCK));
	dim3 block(BLOCK,BLOCK);

	//Time
	//cudaEvent_t start,stop;
	//float elapsedtime;
	//cudaEventCreate(&start);
	//cudaEventRecord(start,0);

	//function
	cuda_sub<<<grid,block>>>(dev_c,dev_a,dev_b,width,height);
	checkCudaError(cudaGetLastError());
	cudaDeviceSynchronize();
	checkCudaError(cudaGetLastError());
	
	//Time
	/*cudaEventCreate(&stop);
	cudaEventRecord(stop,0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedtime,start,stop);
	printf("Time spent for calculation in CUDA : %.10f s\n",elapsedtime/(float)1000);*/

	cudastatus=cudaGetLastError();
	if (cudastatus!=cudaSuccess)
		return cudastatus;
	
	//copyback
	cudastatus=cudaMemcpy(c,dev_c,width*height*sizeof(float),cudaMemcpyDeviceToHost);
	cudaFree(dev_a);
	cudaFree(dev_b);
	cudaFree(dev_c);

	return cudastatus;
}

/*int main(int argc, char *argv[]){
	int width,height;
	
	//checking args and getting args
	if(argc<5){
		printf("Please enter all args eg: ./add file1.txt file2.txt rows cols ans.txt");
		exit(1);
		}
		
	//char matf1[]=argv[1];
	width=atoi(argv[4]);
	height=atoi(argv[3]);
	
	//allocating
	float *mat1=(float *)malloc(width*height*sizeof(float));
	isMemoryFull(mat1);
	float *mat2=(float *)malloc(width*height*sizeof(float));
	isMemoryFull(mat2);
	float *ans=(float *)malloc(width*height*sizeof(float));
	isMemoryFull(ans);
	
	//reading files
	int i,j;
	FILE *fp;
	fp=fopen(argv[1],"r");
	isFileOK(fp);
	for (i=0;i<width*height;i++){
		fscanf(fp,"%f",&mat1[i]);
		}
	fclose(fp);
	//printf("reading mat 1 finished\n");
	
	fp=fopen(argv[2],"r");
	isFileOK(fp);
	for (i=0;i<width*height;i++){
		fscanf(fp,"%f",&mat2[i]);
		}	
	fclose(fp);
	//printf("reading mat 2 finished\n");

	//add
	
	clock_t start=clock();
	cudaError_t status=add(ans,mat1,mat2,width,height);
	checkCudaError(status);
	
	clock_t stop=clock();
    double cputime=(double)((stop-start)/(float)CLOCKS_PER_SEC);
    printf("Time for calculation with memory transfer overhead : %1.10f s\n",cputime);
	
	//writing to file
	fp=fopen(argv[5],"w");
	isFileOK(fp);
	for (i=0;i<height;i++){
		for (j=0;j<width;j++){
			fprintf(fp,"%f ",ans[width*i+j]);
		}	
		fprintf(fp,"\n");
	}
	fclose(fp);
	
	return 0;
}*/

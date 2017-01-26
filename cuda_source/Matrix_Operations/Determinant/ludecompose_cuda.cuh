//author : H.M.Gamaarachchi
//c file for LU decomposition using cuda
//reference : http://rosettacode.org/wiki/LU_decomposition#C

#include <stdio.h>
#include <string.h>
#include <assert.h>
#include <stdlib.h>
#include <time.h>
#include "myhelpers.h"

#define BLOCK_SIZE 16

//print an array
void printarray(float *mat,int width,int height){
	int i,j;
	for (i=0;i<height;i++){
		for (j=0;j<width;j++){
			printf("%f ",mat[width*i+j]);
			}
		printf("\n");
	}
}

//finding highest factor <=maxsize
int hf(int size,int maxsize){ 
  int i=1;int factor=1;
  for (i=2;i<=maxsize;i++){
	if(size%i==0){
		factor=i;
	}
  }
  return factor;
}

//initialize matrices
__global__ void initialize(float *matL,float *matU,int size){
	int i,j;
	
	i= blockIdx.y * blockDim.y + threadIdx.y;
	j= blockIdx.x * blockDim.x + threadIdx.x;

	if(i<size && j<size){
		//make matL to an identity matrix and matU to a zero matrix
		matU[i*size+j]=0;
		matL[i*size+j]=0;
		if (i==j){
			matL[i*size+j]=1;
		}
	}


}

//do LU decomposition in parallel nth row nth col in 2D array (inefficient)
__global__ void LUdecompose_CUDA_test(float *matL,float *matU,float *matA,int size,int current){

	int i,j,k;
	float u,l;
	j= blockIdx.x * blockDim.x + threadIdx.x;
	i= blockIdx.y * blockDim.y + threadIdx.y;
	
	if(i==current || j==current && i<size && j<size){
		
		u=matA[i*size+j]; //a(i,j)
		l=matA[i*size+j];		
		//calculating uppers and lowers
		if (j>=i){
			for(k=0;k<i;k++){
				u=u-matU[k*size+j]*matL[i*size+k]; //u(i,j)=a(i,j)-sigma(u(k,j)*l(i,k))
				}
			matU[i*size+j]=u; //substitute calculated uppers
		}
		else if (j<=i){
			for(k=0;k<j;k++){
				l=l-matU[k*size+j]*matL[i*size+k];	
			}
			matL[i*size+j]=l/(float)matU[j*size+j];  //substitute calculated lowers
		}

	}
	
			
}	

//do LU decomposition in parallel nth row nth col using 1 dimensional thread array
__global__ void LUdecompose_CUDA(float *matL,float *matU,float *matA,int size,int current){

	int i,j,k,n;
	float u,l;
	n= blockIdx.x * blockDim.x + threadIdx.x;
	
	if(n>=current && n<size){
	
		//calculating uppers in ith row
		i=current; 
		j=n;
		u=matA[i*size+j]; //a(i,j)
		l=matA[i*size+j];	
		if (j>=i){
			for(k=0;k<i;k++){
				u=u-matU[k*size+j]*matL[i*size+k]; //u(i,j)=a(i,j)-sigma(u(k,j)*l(i,k))
				}
			matU[i*size+j]=u; //substitute calculated uppers
		}

		//calculating lowers ith colum
		i=n;
		j=current;
		u=matA[i*size+j]; //a(i,j)
		l=matA[i*size+j];	
		if (j<=i){
			for(k=0;k<j;k++){
				l=l-matU[k*size+j]*matL[i*size+k];	
			}
			matL[i*size+j]=l/(float)matU[j*size+j];  //substitute calculated lowers
		}
	}

	
	
			
}	


//LU decomposition abstraction	
void LUdecompose(float *matL,float *matU,float *matA,int size){
	float *dev_L,*dev_U,*dev_A;
	
	//allocating memory on device
	checkCudaError(cudaMalloc((void**)&dev_L, size*size*sizeof(float)));
	checkCudaError(cudaMalloc((void**)&dev_U, size*size*sizeof(float)));
	checkCudaError(cudaMalloc((void**)&dev_A, size*size*sizeof(float)));
	
	//Copying data to device
	checkCudaError(cudaMemcpy(dev_A,matA, size*size*sizeof(float),cudaMemcpyHostToDevice));
	
	//thread distribution for initialization
	dim3 grid(ceil(size/(float)BLOCK_SIZE),ceil(size/(float)BLOCK_SIZE));
	dim3 block(BLOCK_SIZE,BLOCK_SIZE);

	//thread distribution for calculation
	int block1=hf(size,32);
	int grid1=ceil(size/block1);

	//Time
	cudaEvent_t start,stop;
	float elapsedtime;
	cudaEventCreate(&start);
	cudaEventRecord(start,0);	

	//function
	initialize<<<grid,block>>>(dev_L,dev_U, size);
	int i=0;
	for (i=0;i<size;i++){
		LUdecompose_CUDA<<<grid1,block1>>>(dev_L,dev_U,dev_A,size,i);
		cudaDeviceSynchronize();
	}

	//Time
	cudaEventCreate(&stop);
	cudaEventRecord(stop,0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedtime,start,stop);
	printf("Time spent for calculation in CUDA : %.10f s\n",elapsedtime/(float)1000);
	
	//copyback
	checkCudaError(cudaMemcpy(matL,dev_L,size*size*sizeof(float),cudaMemcpyDeviceToHost));
	checkCudaError(cudaMemcpy(matU,dev_U,size*size*sizeof(float),cudaMemcpyDeviceToHost));
	cudaFree(dev_A);
	cudaFree(dev_L);
	cudaFree(dev_U);

}


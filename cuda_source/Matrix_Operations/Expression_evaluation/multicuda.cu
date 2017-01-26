/*
Author : H.M.Gamaarachchi
C file for multiplication of 2 matrices in CUDA using global memory
*/


/*#include <stdio.h>
#include <string.h>
#include <assert.h>
#include <stdlib.h>
#include "myhelpers.h"

#define BLOCK 16*/

/*print an array
void printarray(float *mat,int width,int height){
	int i,j;
	for (i=0;i<height;i++){
		for (j=0;j<width;j++){
			printf("%f ",mat[width*i+j]);
			}
		printf("\n");
	}
}
*/

//do multiplication in cuda
__global__ void multi_cuda(float *ans,float *mat1,int width1,int height1,float *mat2,int width2,int height2){
	int i,j,k;
	float sum;
	i=blockDim.y*blockIdx.y+threadIdx.y;
	j=blockDim.x*blockIdx.x+threadIdx.x;
	
	if(i<height1 && j<width2){
		sum=0;
			for (k=0;k<width1;k++){
				sum=sum+mat1[width1* i+ k]*mat2[width2* k + j];
			}
		ans[width2*i+j]=sum;
	}

}	


//do multiplication abstraction
void multi(float *ans,float *mat1,int width1,int height1,float *mat2,int width2,int height2){
	
	//allocating
	float *dev_mat1,*dev_mat2,*dev_ans;
	checkCudaError(cudaMalloc((void**)&dev_mat1,sizeof(float)*width1*height1));
	checkCudaError(cudaMalloc((void**)&dev_mat2,sizeof(float)*width2*height2));
	checkCudaError(cudaMalloc((void**)&dev_ans,sizeof(float)*width2*height1));

	//copying
	checkCudaError(cudaMemcpy(dev_mat1,mat1,sizeof(float)*width1*height1,cudaMemcpyHostToDevice));
	checkCudaError(cudaMemcpy(dev_mat2,mat2,sizeof(float)*width2*height2,cudaMemcpyHostToDevice));

//time calculations
	//cudaEvent_t start,stop;
	//float elapsedtime;
	//cudaEventCreate(&start);
	//cudaEventRecord(start,0);

	//calling cuda func
	dim3 grid(ceil(width2/(float)BLOCK),ceil(height1/(float)BLOCK));
	dim3 block(BLOCK,BLOCK);
	multi_cuda<<<grid,block>>>(dev_ans,dev_mat1,width1,height1,dev_mat2,width2,height2);
	checkCudaError(cudaGetLastError());
	cudaDeviceSynchronize();
	checkCudaError(cudaGetLastError());

//time calculation
	/*cudaEventCreate(&stop);
	cudaEventRecord(stop,0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedtime,start,stop);
	printf("Time spent for calculation in CUDA : %.10f s\n",elapsedtime/(float)1000);*/

	//copy back
	checkCudaError(cudaMemcpy(ans,dev_ans,sizeof(float)*width2*height1,cudaMemcpyDeviceToHost));
	cudaFree(dev_mat1);
	cudaFree(dev_mat2);
	cudaFree(dev_ans);
}	


/*
int main(int argc, char *argv[]){
	int width1,width2,height1,height2;
	
	//checking args and getting args
	if(argc<8){
		printf("Please enter all args eg: ./multi file1.txt rows1 cols1 file2.txt rows2 cols2 ans.txt");
		exit(1);
		}
		
	//char matf1[]=argv[1];
	width1=atoi(argv[3]);
	height1=atoi(argv[2]);
	//char matf2[]=argv[4];
	width2=atoi(argv[6]);
	height2=atoi(argv[5]);
	
	//check dims
	if (width1!=height2){
		printf("Please recheck the dimensions.. cols_matA must be equal to rows_matB\n");
		exit(EXIT_FAILURE);
		}
		
	//allocating
	float *mat1=(float *)malloc(width1*height1*sizeof(float));
	isMemoryFull(mat1);
	float *mat2=(float *)malloc(width2*height2*sizeof(float));
	isMemoryFull(mat2);
	float *ans=(float *)malloc(width2*height1*sizeof(float));
	isMemoryFull(ans);
	
	//reading files
	int i,j;
	FILE *fp;
	fp=fopen(argv[1],"r");
	isFileOK(fp);
	for (i=0;i<width1*height1;i++){
		fscanf(fp,"%f",&mat1[i]);
		}
	fclose(fp);

	//printf("reading mat 1 finished\n");
	
	fp=fopen(argv[4],"r");
	isFileOK(fp);
	for (i=0;i<width2*height2;i++){
		fscanf(fp,"%f",&mat2[i]);
		}	
	fclose(fp);
	
	//printf("reading mat 2 finished\n");

	// multi
	clock_t start=clock();
	
	multi(ans,mat1,width1,height1,mat2,width2,height2);
	checkCudaError(cudaGetLastError());
	
	clock_t stop=clock();
    double cputime=(double)((stop-start)/(float)CLOCKS_PER_SEC);
    printf("Time for calculation with memory transfer overhead : %1.10f s\n",cputime);
	
	//writing to file
	fp=fopen(argv[7],"w");
	isFileOK(fp);
	for (i=0;i<height1;i++){
		for (j=0;j<width2;j++){
			fprintf(fp,"%f ",ans[width2*i+j]);
		}	
		fprintf(fp,"\n");
	}
	fclose(fp);
	
	free(mat1);
	free(mat2);
	free(ans);

	return 0;
}*/

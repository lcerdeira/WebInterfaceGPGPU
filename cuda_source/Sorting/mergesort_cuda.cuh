
/* CUDA C file for merge sort
Author : H.M.Gamaarachchi
*/

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <string.h>
#include <time.h>
#include <stdlib.h>
#include "myhelpers.h"

#define BLOCK 1024

/**************************ascending*******************************************/

/*merging operation for CPU
 * l:leftmost index   r: rightmost index   q:index of 1st element of list 2*/
void merge_asc(float list[],int l,int q,int r){ 
	
	float *temp=(float *)malloc(sizeof(int)*(r-l+1)); //temperory list
	int i=l,j=q,n=0;
	
	// appending to new list least values from the sorted lists
	while(i<q && j<=r){
		if (list[i]<=list[j]){
			temp[n]=list[i];
			n++;i++;
		}
		else{
			temp[n]=list[j];
			n++;j++;
		}
	}
	
	//appending remaining items in lists after comparisons
	while (i<q){
		temp[n]=list[i];
		i++;n++;
	}
	while (j<=r){
		temp[n]=list[j];
		j++;n++;
	}
	
	//copying sorted items to the original list
	for (i=l,j=0;i<=r;i++,j++){
		list[i]=temp[j];
	}
	
	free(temp);
	return ;
}


/*non recursive function for merge sort for CPU*/
void cpu_mergeSort_asc(float list[],int n,int step){
	int i=0,l,q,r;
	while(step<n-1){
		i=0;
		while(i+2*step<=n){
			l=i;
			q=i+step;
			r=i+2*step-1;
			merge_asc(list,l,q,r);
			i=r+1;
		}
		if (r<n-1)
			merge_asc(list,l,r+1,n-1);
		step=step*2;
	}
	return;
	}
	


//ascending kernel will sort upto blocks given in BLOCK
__global__ void cuda_sort_asc(float *data, int size){

	int x=blockIdx.x * blockDim.x + threadIdx.x;
	float temp[BLOCK];
	int step=1;

		//i is the marker for left side, j is the marger for right list, n is the marker for the temp 
		//l is the left most index of the left list r is the rightmost index of the right list q is the leftmost ondex of the right list
		int i=0,j=0,n=0,l,r,q;
	
	//step is the size of a sub list to be merged
	//while(step<=size-1){
	while(step<=BLOCK-1 && step<=size-1){

		if (x%(step*2)==0 && x<size){ //selecting only correct threads

			l= blockIdx.x * blockDim.x + threadIdx.x ; //left of left
			r= l+2*step-1; //right of right
			if (r>=size){
				r=size-1; //incase right is out of size
			}
			q =l+step; //left of right
			if (q>=size){
				q=size-1;
			}
		
			//marker setting
			i=l;j=q;n=0;

			// appending to new list least values from the sorted lists
			while(i<q && j<=r){
				if (data[i]<=data[j]){
					temp[n]=data[i];
					i++;
				}
				else {
					temp[n]=data[j];
					j++;
				}
				n++;
			}

	
			//appending remaining items in lists after comparisons
			while (i<q){
				temp[n]=data[i];
				i++;n++;
			}
			while (j<=r){
				temp[n]=data[j];
				j++;n++;
			}
			//copying sorted items to the original list
			for (i=l,j=0;i<=r;i++,j++){
				data[i]=temp[j];
			}

		}
			
		step=step*2;
		__syncthreads();

	}
	

}

/****************************************Descending************************************/

/*merging operation for CPU
 * l:leftmost index   r: rightmost index   q:index of 1st element of list 2*/
void merge_desc(float list[],int l,int q,int r){ 
	
	float *temp=(float *)malloc(sizeof(int)*(r-l+1)); //temperory list
	int i=l,j=q,n=0;
	
	// appending to new list least values from the sorted lists
	while(i<q && j<=r){
		if (list[i]>=list[j]){
			temp[n]=list[i];
			n++;i++;
		}
		else{
			temp[n]=list[j];
			n++;j++;
		}
	}
	
	//appending remaining items in lists after comparisons
	while (i<q){
		temp[n]=list[i];
		i++;n++;
	}
	while (j<=r){
		temp[n]=list[j];
		j++;n++;
	}
	
	//copying sorted items to the original list
	for (i=l,j=0;i<=r;i++,j++){
		list[i]=temp[j];
	}
	
	free(temp);
	return ;
}


/*non recursive function for merge sort for CPU*/
void cpu_mergeSort_desc(float list[],int n,int step){
	int i=0,l,q,r;
	while(step<n-1){
		i=0;
		while(i+2*step<=n){
			l=i;
			q=i+step;
			r=i+2*step-1;
			merge_desc(list,l,q,r);
			i=r+1;
		}
		if (r<n-1)
			merge_desc(list,l,r+1,n-1);
		step=step*2;
	}
	return;
	}
	


//ascending kernel will sort upto blocks given in BLOCK
__global__ void cuda_sort_desc(float *data, int size){

	int x=blockIdx.x * blockDim.x + threadIdx.x;
	float temp[BLOCK];
	int step=1;

		//i is the marker for left side, j is the marger for right list, n is the marker for the temp 
		//l is the left most index of the left list r is the rightmost index of the right list q is the leftmost ondex of the right list
		int i=0,j=0,n=0,l,r,q;
	
	//step is the size of a sub list to be merged
	//while(step<=size-1){
	while(step<=BLOCK-1 && step<=size-1){

		if (x%(step*2)==0 && x<size){ //selecting only correct threads

			l= blockIdx.x * blockDim.x + threadIdx.x ; //left of left
			r= l+2*step-1; //right of right
			if (r>=size){
				r=size-1; //incase right is out of size
			}
			q =l+step; //left of right
			if (q>=size){
				q=size-1;
			}
		
			//marker setting
			i=l;j=q;n=0;

			// appending to new list least values from the sorted lists
			while(i<q && j<=r){
				if (data[i]>=data[j]){
					temp[n]=data[i];
					i++;
				}
				else {
					temp[n]=data[j];
					j++;
				}
				n++;
			}

	
			//appending remaining items in lists after comparisons
			while (i<q){
				temp[n]=data[i];
				i++;n++;
			}
			while (j<=r){
				temp[n]=data[j];
				j++;n++;
			}
			//copying sorted items to the original list
			for (i=l,j=0;i<=r;i++,j++){
				data[i]=temp[j];
			}

		}
			
		step=step*2;
		__syncthreads();

	}
	

}

/***************************************sorting********************************************/
// sort asbstraction for cuda
void sort(float *sorted,float *data,int size, int order){
	//cuda memeory allocation and copy
	float *dev_data;
	checkCudaError(cudaMalloc((void**)&dev_data,sizeof(float)*size));
	checkCudaError(cudaMemcpy(dev_data,data,sizeof(float)*size,cudaMemcpyHostToDevice));
	int grid=ceil(size/(float)BLOCK);

	cudaEvent_t start,stop;
	float elapsed_time;
	cudaEventCreate(&start);
	cudaEventRecord(start,0);

	//cuda sort calling
	if (order==1){
		cuda_sort_asc<<<grid,BLOCK>>>(dev_data,size);
		checkCudaError(cudaGetLastError());
		cudaDeviceSynchronize();
	}
	else if (order==2){
		cuda_sort_desc<<<grid,BLOCK>>>(dev_data,size);
		checkCudaError(cudaGetLastError());
		cudaDeviceSynchronize();
	}
	else{
		printf("only 1 or 2 are valid args for the order");
		exit(1);
	}

	cudaEventCreate(&stop);
	cudaEventRecord(stop,0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsed_time,start,stop);
	printf("Time spent for sorting upto %d size blocks in CUDA is %f s\n",BLOCK,elapsed_time/(float)1000);

	//copyback and free
	checkCudaError(cudaMemcpy(sorted,dev_data,sizeof(float)*size,cudaMemcpyDeviceToHost));
	
	//sorting rest in CPU
	if (order==1){
		cpu_mergeSort_asc(sorted,size,BLOCK);
	}
	else if (order==2){
		//cpu_mergeSort_desc(sorted,size,BLOCK);
	}
		
	cudaFree(dev_data);	
}


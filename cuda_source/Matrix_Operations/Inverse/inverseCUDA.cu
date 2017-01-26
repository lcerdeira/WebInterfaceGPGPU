/*author : H.M.Gamaarachchi
c file for solving linear equations using LU decomposition
reference : http://rosettacode.org/wiki/LU_decomposition#C
http://www.ece.mcmaster.ca/~kiruba/3sk3/lecture6.pdf
 A.x[i th col]=b[ith col] solved by forward substitution L.d[ith col]=b[ith col] and the backward substitution U.x[ith col]=d[ith col]
*/

#include "ludecompose_cuda.cuh"

__global__ void initialize1(float *matL,float *matU,float *matb, int size){
	int i,j;
	
	i= blockIdx.y * blockDim.y + threadIdx.y;
	j= blockIdx.x * blockDim.x + threadIdx.x;

	//make matL to an identity matrix and matU to a zero matrix
	matU[i*size+j]=0;
	matL[i*size+j]=0;
	matb[i*size+j]=0;
	if (i==j){
		matL[i*size+j]=1;
		matb[i*size+j]=1;
	}


}
//linear solving to get inverse
__global__ void linearSolve(float *inverse, float *matL, float *matU,float *matb,float *tempd, int size){
	int i,j,col_mat;
	float d,x;
	
	col_mat = blockIdx.x * blockDim.x + threadIdx.x;

		//forward substitution
		for (i=0;i<size;i++){
			d=matb[i*size+col_mat]; //initially di=bi
			for (j=0;j<i;j++){
				d=d-matL[i*size+j]*tempd[j*size+col_mat]; //di=bi-sigma(Lij*dj)
			}
			tempd[i*size+col_mat]=d;
		}
		
		//backward substitution
		for (i=size-1;i>=0;i--){
			x=tempd[i*size+col_mat]; //initially xi=di
			for (j=i+1;j<size;j++){
				x=x-matU[i*size+j]*inverse[j*size+col_mat]; //xi=di-sigma(Uij*xj)
			}
			inverse[i*size+col_mat]=x/matU[i*size+i]; //xi=xi/Uii
		}	

}

//Inversion abstraction	
void inverse(float *inverse,float *matA,int size){
	float *dev_L,*dev_U,*dev_A,*dev_b,*dev_d,*dev_inverse;
	
	//allocating memory on device
	checkCudaError(cudaMalloc((void**)&dev_A, size*size*sizeof(float))); //original matrix to find inverse
	checkCudaError(cudaMalloc((void**)&dev_L, size*size*sizeof(float)));
	checkCudaError(cudaMalloc((void**)&dev_U, size*size*sizeof(float)));
	checkCudaError(cudaMalloc((void**)&dev_b, size*size*sizeof(float)));	
	checkCudaError(cudaMalloc((void**)&dev_d, size*size*sizeof(float)));
	checkCudaError(cudaMalloc((void**)&dev_inverse, size*size*sizeof(float)));

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
	initialize1<<<grid,block>>>(dev_L,dev_U,dev_b, size); //initialization
	checkCudaError(cudaGetLastError());
	cudaDeviceSynchronize();
	
	int i=0;
	for (i=0;i<size;i++){
		//decomposition
		LUdecompose_CUDA<<<grid1,block1>>>(dev_L,dev_U,dev_A,size,i);
		cudaDeviceSynchronize();
	}
	checkCudaError(cudaGetLastError());
	
	linearSolve<<<grid1,block1>>>(dev_inverse, dev_L, dev_U, dev_b, dev_d,size); //solving linear equations
	checkCudaError(cudaGetLastError());
	cudaDeviceSynchronize();

	//Time
	cudaEventCreate(&stop);
	cudaEventRecord(stop,0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedtime,start,stop);
	printf("Time spent for calculation in CUDA : %.10f s\n",elapsedtime/(float)1000);
	
	//copyback
	checkCudaError(cudaMemcpy(inverse,dev_inverse,size*size*sizeof(float),cudaMemcpyDeviceToHost));

	cudaFree(dev_A);
	cudaFree(dev_L);
	cudaFree(dev_U);
	cudaFree(dev_b);
	cudaFree(dev_d);
	cudaFree(dev_inverse);

}

int main(int argc, char *argv[]){
	int size;
	
	//checking args and getting args
	if(argc<4){
		fprintf(stderr,"Please enter all args eg: ./Inverse mat.txt rows ans.txt");
		exit(EXIT_FAILURE);
		}
		
	//char matf1[]=argv[1];
	size=atoi(argv[2]);

	//allocating
	float *matA=(float *)malloc(size*size*sizeof(float));
	isMemoryFull(matA);
	float *ans=(float *)malloc(size*size*sizeof(float));
	isMemoryFull(ans);
	
	//reading files
	int i,j;
	FILE *fp;
	fp=fopen(argv[1],"r");
	isFileOK(fp);
	for (i=0;i<size*size;i++){
		fscanf(fp,"%f",&matA[i]);
		}
	fclose(fp);
	//printf("reading matrix finished\n");

	// Inverse
	clock_t start=clock();

	inverse(ans,matA,size);
	checkCudaError(cudaGetLastError());
	
	clock_t stop=clock();
    double cputime=(double)((stop-start)/(float)CLOCKS_PER_SEC);
    printf("Time for calculation with memory transfer overhead : %1.10f s\n",cputime);
	
	//writing to file
	fp=fopen(argv[3],"w");
	isFileOK(fp);
	for (i=0;i<size;i++){
		for (j=0;j<size;j++){
			fprintf(fp,"%f ",ans[size*i+j]);
		}	
		fprintf(fp,"\n");;
	}
	fclose(fp);
	
	free(matA);
	free(ans);
	
	return 0;
}

//author : H.M.Gamaarachchi
//c file for finding determinant using cuda
//reference : http://rosettacode.org/wiki/LU_decomposition#C

#include "ludecompose_cuda.cuh"

//determinant
float determinant(float *matL,float *matU,int size){
	int i;
	float det=1;
	for  (i=0;i<size;i++){
		det=det*matL[i*size+i]*matU[i*size+i];
	}
	return det;
}

int main(int argc, char *argv[]){
	int size;
	
	//checking args and getting args
	if(argc<4){
		fprintf(stderr,"Please enter all args eg: ./det mat.txt rows ans.txt");
		exit(EXIT_FAILURE);
		}
		
	//char matf1[]=argv[1];
	size=atoi(argv[2]);

	//allocating
	float *matA=(float*)malloc(size*size*sizeof(float));
	isMemoryFull(matA);
	float *matL=(float*)malloc(size*size*sizeof(float));
	isMemoryFull(matL);
	float *matU=(float*)malloc(size*size*sizeof(float));
	isMemoryFull(matU);
	
	//reading files
	int i;
	FILE *fp;
	fp=fopen(argv[1],"r");
	isFileOK(fp);
	for (i=0;i<size*size;i++){
		fscanf(fp,"%f",&matA[i]);
		}
	fclose(fp);
	//printf("reading mat A finished\n");

	// LU decompose
	clock_t start=clock();
	
	LUdecompose(matL,matU,matA,size);
	checkCudaError(cudaGetLastError());
	float det=determinant(matL,matU,size);
	
	clock_t stop=clock();
    double cputime=(double)((stop-start)/(float)CLOCKS_PER_SEC);
	printf("Time for calculation with memory transfer overhead : %1.10f s\n",cputime);
	
	//writing to file
	
	fp=fopen(argv[3],"w");
	isFileOK(fp);
	fprintf(fp,"%f",det);
	fclose(fp);

	free(matA);
	free(matU);
	free(matL);
	
	
	return 0;
}

//author : H.M.Gamaarachchi
//c file for LU decomposition using cuda
//reference : http://rosettacode.org/wiki/LU_decomposition#C

#include "ludecompose_cuda.cuh"

int main(int argc, char *argv[]){
	int size;
	
	//checking args and getting args
	if(argc<5){
		fprintf(stderr,"Please enter all args eg: ./LUdecompose file1.txt rows L.txt U.txt");
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
	int i,j;
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
	
	clock_t stop=clock();
    double cputime=(double)((stop-start)/(float)CLOCKS_PER_SEC);
    printf("Time for calculation with memory transfer overhead : %1.10f s\n",cputime);
	
	//writing to file
	FILE *fp1;
	fp=fopen(argv[3],"w");
	isFileOK(fp);
	fp1=fopen(argv[4],"w");
	isFileOK(fp1);
	for (i=0;i<size;i++){
		for (j=0;j<size;j++){
			fprintf(fp,"%f ",matL[size*i+j]);
			fprintf(fp1,"%f ",matU[size*i+j]);
		}	
		fprintf(fp,"\n");
		fprintf(fp1,"\n");
	}
	fclose(fp);
	fclose(fp1);
	
	free(matA);
	free(matL);
	free(matU);
	
	return 0;
}
